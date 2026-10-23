#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vidar — blescanner.py  v5.3
Raspberry Pi 5 · Sony IMX500 AI Camera build

Changes from v5.2:
  - EmployeeRegistry now keys on normalised beacon MAC (not uuid/major/minor tuple)
  - sync-employees call now includes camera_id in body so Edge Fn can auth + resolve store_id
  - on_advertisement looks up by device.address (BLE MAC) instead of iBeacon UUID
  - sync-employees and write-ble-pings both use Authorization header only (no token in body)
"""

import asyncio
import json
import logging
import signal
import stat
import struct
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import requests
from bleak import BleakScanner
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData

# ─── PATHS ────────────────────────────────────────────────────────────────────

CONFIG_PATH           = Path.home() / "retail" / "config.json"
BLE_STATE_PATH        = Path.home() / "retail" / "ble_state.json"
EMPLOYEES_CACHE_PATH  = Path.home() / "retail" / "employees_cache.json"
EMPLOYEES_RELOAD_FLAG = Path.home() / "retail" / "employees_updated.flag"

IST = timezone(timedelta(hours=5, minutes=30))

# ─── CONFIG LOAD ──────────────────────────────────────────────────────────────

def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.json not found at {CONFIG_PATH}")
    mode = CONFIG_PATH.stat().st_mode
    if mode & (stat.S_IRGRP | stat.S_IROTH):
        print(f"WARNING: {CONFIG_PATH} is readable by group/others. Run: chmod 600 {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    required = ["camera_id", "device_token", "org_id", "store_id",
                "edge_function_url", "ble_pings_edge_url"]
    for k in required:
        if k not in cfg:
            raise KeyError(f"config.json missing required key: {k!r}")
    return cfg

CONFIG             = _load_config()
CAMERA_ID          = CONFIG["camera_id"]
DEVICE_TOKEN       = CONFIG["device_token"]
ORG_ID             = CONFIG["org_id"]
STORE_ID           = CONFIG["store_id"]
EDGE_URL           = CONFIG["edge_function_url"].rstrip("/")
BLE_PINGS_EDGE_URL = CONFIG["ble_pings_edge_url"]
SYNC_EMPLOYEES_URL = f"{EDGE_URL}/sync-employees"

# Token only goes in the Authorization header — never in JSON body
_DEVICE_HEADERS = {
    "Authorization": f"Bearer {DEVICE_TOKEN}",
    "Content-Type":  "application/json",
}

# ─── LOGGING ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BLE] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("blescanner")

# ─── TUNING ───────────────────────────────────────────────────────────────────

BLE_ABSENT_SECS          = 300
EMPLOYEE_REFRESH_SECS    = 3600
SCAN_INTERVAL_SECS       = 2.0
PERIODIC_FLUSH_INTERVAL  = 1800
SUMMARY_INTERVAL_SECS    = 60
SHUTDOWN_FLUSH_WAIT_SECS = 8

# ─── GRACEFUL SHUTDOWN ────────────────────────────────────────────────────────

_shutdown = threading.Event()

def _handle_signal(signum, frame):
    log.info(f"Signal {signum} received — shutting down")
    _shutdown.set()

signal.signal(signal.SIGTERM, _handle_signal)

# ─── EDGE FUNCTION CALLER ─────────────────────────────────────────────────────

def call_edge(url: str, payload: dict, timeout: int = 15) -> object:
    """
    POST to a Supabase Edge Function.
    Token is only in the Authorization header — never in the payload.
    Retries once on transient failure.
    """
    for attempt in range(2):
        try:
            r = requests.post(url, headers=_DEVICE_HEADERS, json=payload, timeout=timeout)
            if not r.ok:
                log.error(f"Edge fn {url} → HTTP {r.status_code}: {r.text[:300]}")
                r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            if attempt == 0:
                log.warning(f"Edge fn call failed, retrying in 3s: {exc}")
                time.sleep(3)
            else:
                log.error(f"Edge fn call failed after retry: {exc}")
                raise
    return {}

# ─── UUID / MAC NORMALISATION ─────────────────────────────────────────────────

def _norm_mac(raw: str) -> str:
    """Normalise a MAC or UUID string: lowercase, strip separators."""
    return raw.replace("-", "").replace(":", "").lower()

# ─── EMPLOYEE REGISTRY ────────────────────────────────────────────────────────

class EmployeeRegistry:
    """
    Fetches the active employee roster via the sync-employees Edge Function.
    Keys on normalised beacon MAC address for O(1) lookup.
    Writes employees_cache.json atomically so detector.py can hot-reload.
    """

    def __init__(self):
        self._lock: threading.Lock = threading.Lock()
        self._lookup: dict = {}   # normalised_mac → employee dict
        self._load()

    def _load(self) -> None:
        try:
            # camera_id is required so the Edge Function can auth + resolve store_id
            rows = call_edge(SYNC_EMPLOYEES_URL, {"camera_id": CAMERA_ID})
            if not isinstance(rows, list):
                raise ValueError(f"sync-employees returned non-list: {type(rows)}")
            m = {}
            for row in rows:
                mac = row.get("beacon_mac") or ""
                if not mac:
                    log.warning(f"Employee {row.get('id')} has no beacon_mac — skipping")
                    continue
                key = _norm_mac(mac)
                m[key] = {
                    "employee_id": row["id"],
                    "name":        row.get("name", ""),
                    "gender":      row.get("gender", "unknown"),
                    "age_bracket": row.get("age_bracket", "unknown"),
                    "beacon_mac":  mac,
                }
            with self._lock:
                self._lookup = m
            log.info(f"Employee registry: {len(m)} active employee(s) loaded")
            self._write_cache(m)
        except Exception as exc:
            log.error(f"Employee registry load failed: {exc}")

    def _write_cache(self, lookup: dict) -> None:
        """
        Atomically write employees_cache.json and touch the reload flag
        so detector.py picks up the new roster without restarting.
        """
        cache = list(lookup.values())
        tmp   = EMPLOYEES_CACHE_PATH.with_suffix(".tmp")
        try:
            EMPLOYEES_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w") as f:
                json.dump(cache, f, indent=2)
            tmp.rename(EMPLOYEES_CACHE_PATH)
            EMPLOYEES_RELOAD_FLAG.parent.mkdir(parents=True, exist_ok=True)
            EMPLOYEES_RELOAD_FLAG.touch()
            log.info(f"employees_cache.json written: {len(cache)} employee(s)")
        except Exception as exc:
            log.error(f"employees_cache write failed: {exc}")

    def lookup(self, beacon_mac: str) -> Optional[dict]:
        """Look up an employee by their beacon's BLE MAC address."""
        key = _norm_mac(beacon_mac)
        with self._lock:
            return self._lookup.get(key)

    def refresh_loop(self) -> None:
        """Background thread: re-fetch roster every EMPLOYEE_REFRESH_SECS."""
        while not _shutdown.is_set():
            _shutdown.wait(timeout=EMPLOYEE_REFRESH_SECS)
            if _shutdown.is_set():
                break
            log.info("Employee registry: scheduled hourly refresh")
            self._load()


REGISTRY = EmployeeRegistry()

# ─── IBEACON PARSER ───────────────────────────────────────────────────────────

def parse_ibeacon(adv: AdvertisementData) -> Optional[tuple]:
    """Returns (uuid_str, major, minor, tx_power) or None if not an iBeacon."""
    apple = (adv.manufacturer_data or {}).get(0x004C)
    if apple is None or len(apple) < 23:
        return None
    if apple[0] != 0x02 or apple[1] != 0x15:
        return None
    ub       = apple[2:18]
    uuid_str = (f"{ub[0:4].hex()}-{ub[4:6].hex()}-{ub[6:8].hex()}-"
                f"{ub[8:10].hex()}-{ub[10:16].hex()}")
    major    = struct.unpack(">H", apple[18:20])[0]
    minor    = struct.unpack(">H", apple[20:22])[0]
    tx_power = struct.unpack(">b", bytes([apple[22]]))[0]
    return uuid_str, major, minor, tx_power

# ─── HOUR BUCKET ──────────────────────────────────────────────────────────────

def _hour_bucket(utc_now: datetime) -> datetime:
    ist = utc_now.astimezone(IST)
    return ist.replace(minute=0, second=0, microsecond=0).astimezone(timezone.utc)

# ─── PRESENCE TRACKER ─────────────────────────────────────────────────────────

class PresenceTracker:
    """
    Per-employee BLE presence state machine.
    The Edge Function does an UPSERT on (employee_id, hour_bucket) so
    repeated partial flushes for the same hour accumulate counts correctly.
    """

    def __init__(self):
        self._lock  = threading.Lock()
        self._state: dict = {}   # employee_id → state dict

    def see(self, employee_id: str, name: str, beacon_mac: str, rssi: int) -> None:
        mono       = time.monotonic()
        now_utc    = datetime.now(timezone.utc)
        new_bucket = _hour_bucket(now_utc)

        with self._lock:
            if employee_id not in self._state:
                self._state[employee_id] = _new_state(name, beacon_mac, rssi, mono, now_utc, new_bucket)
                log.info(f"Employee arrived: {name}")
                return

            s = self._state[employee_id]

            # Hour boundary — flush old bucket then reset
            if new_bucket != s["bucket"]:
                self._flush_unlocked(employee_id, s)
                _reset_bucket(s, rssi, mono, now_utc, new_bucket)
                return

            # Re-appearance after a gap within the same hour
            gap = mono - s["last_seen_mono"]
            if gap > BLE_ABSENT_SECS:
                win = int((s["last_seen_mono"] - s["win_start_mono"]) / 60)
                s["accum_mins"]    += max(0, win)
                s["win_start_mono"] = mono
                log.info(f"Re-appeared (same hour): {name} gap={gap:.0f}s added={win}min")

            s["last_seen_mono"]  = mono
            s["last_seen_utc"]   = now_utc
            s["ping_count"]     += 1
            s["rssi_sum"]       += rssi
            s["rssi_min"]        = min(s["rssi_min"], rssi)
            s["rssi_max"]        = max(s["rssi_max"], rssi)
            s["last_ping"]       = now_utc

    def check_departures(self) -> None:
        mono = time.monotonic()
        with self._lock:
            gone = [eid for eid, s in self._state.items()
                    if mono - s["last_seen_mono"] > BLE_ABSENT_SECS]
            for eid in gone:
                s = self._state[eid]
                win = int((s["last_seen_mono"] - s["win_start_mono"]) / 60)
                s["accum_mins"] += max(0, win)
                self._flush_unlocked(eid, s)
                log.info(f"Employee departed: {s['name']} accum={s['accum_mins']}min")
                del self._state[eid]

    def flush_all_present(self, reason: str = "periodic") -> list:
        """
        Flush current bucket for all present employees without evicting them.
        Returns threads so callers can join() for clean shutdown.
        """
        mono    = time.monotonic()
        threads = []
        with self._lock:
            for eid, s in self._state.items():
                if mono - s["last_seen_mono"] > BLE_ABSENT_SECS:
                    continue
                win        = int((s["last_seen_mono"] - s["win_start_mono"]) / 60)
                total_mins = s["accum_mins"] + max(0, win)
                n          = s["ping_count"]
                payload    = _build_ping_payload(eid, s, total_mins, n)
                t = threading.Thread(
                    target=_send_ble_pings,
                    args=(payload, f"{s['name']} [{reason}]"),
                    daemon=True,
                )
                t.start()
                threads.append(t)
        if threads:
            log.info(f"BLE flush ({reason}): {len(threads)} employee(s)")
        return threads

    def present_employees(self) -> list:
        mono = time.monotonic()
        with self._lock:
            return [
                {
                    "employee_id": eid,
                    "name":        s["name"],
                    "beacon_mac":  s["beacon_mac"],
                    "rssi":        s["rssi_sum"] // s["ping_count"] if s["ping_count"] else -99,
                }
                for eid, s in self._state.items()
                if mono - s["last_seen_mono"] <= BLE_ABSENT_SECS
            ]

    def _flush_unlocked(self, eid: str, s: dict) -> None:
        """Build payload and fire a daemon thread. Called with lock held."""
        win        = int((s["last_seen_mono"] - s["win_start_mono"]) / 60)
        total_mins = s["accum_mins"] + max(0, win)
        n          = s["ping_count"]
        payload    = _build_ping_payload(eid, s, total_mins, n)
        threading.Thread(
            target=_send_ble_pings,
            args=(payload, s["name"]),
            daemon=True,
        ).start()


def _build_ping_payload(eid: str, s: dict, total_mins: int, n: int) -> dict:
    return {
        "employee_id":    eid,
        "beacon_mac":     s["beacon_mac"],
        "org_id":         ORG_ID,
        "store_id":       STORE_ID,
        "hour_bucket":    s["bucket"].isoformat(),
        "ping_count":     n,
        "avg_rssi":       s["rssi_sum"] // n if n else None,
        "min_rssi":       s["rssi_min"] if n else None,
        "max_rssi":       s["rssi_max"] if n else None,
        "first_ping_at":  s["first_ping"].isoformat(),
        "last_ping_at":   s["last_ping"].isoformat(),
        "minutes_present":total_mins,
    }


def _new_state(name, mac, rssi, mono, utc, bucket) -> dict:
    return {
        "name":           name,
        "beacon_mac":     mac,
        "bucket":         bucket,
        "ping_count":     1,
        "rssi_sum":       rssi,
        "rssi_min":       rssi,
        "rssi_max":       rssi,
        "first_ping":     utc,
        "last_ping":      utc,
        "win_start_mono": mono,
        "accum_mins":     0,
        "last_seen_mono": mono,
        "last_seen_utc":  utc,
    }


def _reset_bucket(s: dict, rssi, mono, utc, new_bucket) -> None:
    s["bucket"]         = new_bucket
    s["ping_count"]     = 1
    s["rssi_sum"]       = rssi
    s["rssi_min"]       = rssi
    s["rssi_max"]       = rssi
    s["first_ping"]     = utc
    s["last_ping"]      = utc
    s["win_start_mono"] = mono
    s["accum_mins"]     = 0
    s["last_seen_mono"] = mono
    s["last_seen_utc"]  = utc


def _send_ble_pings(payload: dict, label: str) -> None:
    try:
        call_edge(BLE_PINGS_EDGE_URL, payload)
        log.info(
            f"BLE pings → {label} | "
            f"bucket={payload['hour_bucket'][:16]} "
            f"pings={payload['ping_count']} "
            f"avg_rssi={payload['avg_rssi']} "
            f"mins={payload['minutes_present']}"
        )
    except Exception as exc:
        log.error(f"BLE pings flush failed for {label}: {exc}")


TRACKER = PresenceTracker()

# ─── BLE STATE FILE ───────────────────────────────────────────────────────────

def write_ble_state() -> None:
    """
    Atomically write current employee presence to ble_state.json.
    detector.py polls this file every BLE_POLL_SECS seconds.

    Schema:
    {
      "count":     <int>,
      "employees": [{"employee_id": "...", "name": "...", "beacon_mac": "...", "rssi": -48}],
      "ts":        "<iso8601 utc>"
    }
    """
    present = TRACKER.present_employees()
    state   = {
        "count":     len(present),
        "employees": present,
        "ts":        datetime.now(timezone.utc).isoformat(),
    }
    tmp = BLE_STATE_PATH.with_suffix(".tmp")
    try:
        BLE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(state, f)
        tmp.rename(BLE_STATE_PATH)
    except Exception as exc:
        log.error(f"ble_state.json write failed: {exc}")

# ─── ADVERTISEMENT CALLBACK ───────────────────────────────────────────────────

def on_advertisement(device: BLEDevice, adv: AdvertisementData) -> None:
    """
    Fires for every BLE advertisement.
    Only processes iBeacons whose BLE MAC matches a registered employee.
    Everything else is silently ignored — no random device tracking.
    """
    # Must be an iBeacon to be considered at all
    parsed = parse_ibeacon(adv)
    if parsed is None:
        return

    # Look up by the device's BLE hardware MAC address
    employee = REGISTRY.lookup(device.address)
    if employee is None:
        return

    rssi = adv.rssi if adv.rssi is not None else -99
    log.debug(
        f"iBeacon matched: {employee['name']} "
        f"mac={device.address} rssi={rssi}"
    )
    TRACKER.see(employee["employee_id"], employee["name"], employee["beacon_mac"], rssi)

# ─── MAIN SCAN LOOP ───────────────────────────────────────────────────────────

async def scan_loop() -> None:
    log.info("BLE scan started — iBeacon employee tracking active")
    last_summary = 0.0
    last_flush   = 0.0

    async with BleakScanner(
        detection_callback=on_advertisement,
        scanning_mode="active",
    ):
        while not _shutdown.is_set():
            await asyncio.sleep(SCAN_INTERVAL_SECS)

            TRACKER.check_departures()
            write_ble_state()

            now = time.monotonic()

            if now - last_flush >= PERIODIC_FLUSH_INTERVAL:
                TRACKER.flush_all_present(reason="periodic")
                last_flush = now

            if now - last_summary >= SUMMARY_INTERVAL_SECS:
                present = TRACKER.present_employees()
                if present:
                    log.info(
                        f"BLE: {len(present)} present — "
                        + ", ".join(f"{e['name']} rssi={e['rssi']}" for e in present)
                    )
                else:
                    log.info("BLE: no employees present")
                last_summary = now


def main() -> None:
    log.info(
        f"Vidar BLE scanner v5.3 | "
        f"org={ORG_ID} | store={STORE_ID} | "
        f"absent_threshold={BLE_ABSENT_SECS}s | "
        f"periodic_flush={PERIODIC_FLUSH_INTERVAL}s"
    )

    # Write empty state immediately so detector.py never reads a stale file
    write_ble_state()

    # Background thread: refresh employee roster hourly
    threading.Thread(target=REGISTRY.refresh_loop, daemon=True).start()

    try:
        asyncio.run(scan_loop())
    except KeyboardInterrupt:
        _shutdown.set()

    # ── Clean shutdown flush ───────────────────────────────────────────────────
    log.info("Shutting down — flushing BLE presence data...")
    threads = TRACKER.flush_all_present(reason="shutdown")
    deadline = time.monotonic() + SHUTDOWN_FLUSH_WAIT_SECS
    for t in threads:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        t.join(timeout=remaining)
    log.info("BLE scanner stopped.")


if __name__ == "__main__":
    main()



