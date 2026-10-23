#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vidar — detector.py  v2.6
Raspberry Pi 5 · Sony IMX500 AI Camera · MiVolo age/gender

Changes from v2.4/v2.5:
  - Capture command poll loop: polls the `poll-commands` edge function every
    COMMAND_POLL_SECS for pending 'capture' commands for this device's store.
    On a capture command, grabs the latest frame and POSTs it to the
    `capture-snapshot` edge function (with command_id) which uploads to the
    snapshots bucket and closes the device_commands row so the app can preview it.
  - Auth matches the other edge functions: poll-commands and capture-snapshot are
    sent {camera_id, token} in the body; the function SHA-256-hashes the token and
    verifies against the devices table (server resolves store_id — never trusts payload).
  - Added base64 import; _handle_capture() and _command_poll_loop(); thread started in main().

Changes from v2.3 (carried forward):
  - Web preview server (Flask MJPEG + stats JSON) on port 8080.
  - _shared holds "annotated_frame" (BGR JPEG bytes) for the web stream + capture.
"""

import base64          # ─── NEW (capture) ───
import json
import logging
import os
import queue
import signal
import sqlite3
import stat
import threading
import time
import uuid
import warnings
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import argparse
import cv2
import numpy as np
import requests
import schedule

try:
    from flask import Flask, Response, jsonify
    FLASK_OK = True
except ImportError:
    FLASK_OK = False

warnings.filterwarnings("ignore")

# ─── PATHS & CONSTANTS ────────────────────────────────────────────────────────

CONFIG_PATH           = Path.home() /"my_project" / "config.json"
DB_PATH               = Path.home() / "retail" / "vidar.db"
BLE_STATE_PATH        = Path.home() / "retail" / "ble_state.json"
EMPLOYEES_CACHE_PATH  = Path.home() / "retail" / "employees_cache.json"
EMPLOYEES_RELOAD_FLAG = Path.home() / "retail" / "employees_updated.flag"
ZONE_CACHE_PATH       = Path.home() / "retail" / "zone_cache.json"

IMX500_MODEL = "/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk"

IST = timezone(timedelta(hours=5, minutes=30))

# ─── CONFIG LOAD ──────────────────────────────────────────────────────────────

def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.json not found at {CONFIG_PATH}. Run setup first.")
    mode = CONFIG_PATH.stat().st_mode
    if mode & (stat.S_IRGRP | stat.S_IROTH):
        print(f"WARNING: {CONFIG_PATH} is world/group-readable. Run: chmod 600 {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    for k in ["camera_id", "device_token", "org_id", "store_id", "edge_function_url"]:
        if k not in cfg:
            raise KeyError(f"config.json missing required key: {k!r}")
    return cfg

CONFIG        = _load_config()
CAMERA_ID     = CONFIG["camera_id"]
DEVICE_TOKEN  = CONFIG["device_token"]
ORG_ID        = CONFIG["org_id"]
STORE_ID      = CONFIG["store_id"]
EDGE_URL      = CONFIG["edge_function_url"].rstrip("/")
MIVOLO_PATH   = CONFIG.get("mivolo_model",
                    str(Path.home() / "models" / "mivolo_checkpoint.pth.tar"))
YOLO_PATH     = CONFIG.get("yolo_model",
                    str(Path.home() / "models" / "yolov8n.pt"))

# Upload schedule in UTC — override in config.json as "upload_times_utc": ["HH:MM", ...]
UPLOAD_TIMES_UTC: list = CONFIG.get("upload_times_utc", ["03:30", "08:30", "14:30"])

# Web preview port — override in config.json as "preview_port": 8080
PREVIEW_PORT: int = int(CONFIG.get("preview_port", 8080))

_DEVICE_HEADERS = {
    "Authorization": f"Bearer {DEVICE_TOKEN}",
    "Content-Type":  "application/json",
}

# ─── LOGGING ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [DET] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("detector")

# Suppress noisy third-party debug output
for _noisy in ("picamera2", "PIL", "bleak", "urllib3", "httpx",
               "ultralytics", "torch", "timm"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ─── GRACEFUL SHUTDOWN ────────────────────────────────────────────────────────

_shutdown = threading.Event()

def _handle_signal(signum, frame):
    log.info(f"Signal {signum} received — shutting down")
    _shutdown.set()

signal.signal(signal.SIGTERM, _handle_signal)

# ─── CLI ARGS ─────────────────────────────────────────────────────────────────

_parser = argparse.ArgumentParser(description="Vidar detector")
_parser.add_argument(
    "--preview",
    action="store_true",
    help="Open OpenCV preview window (requires a connected display)",
)
_args, _ = _parser.parse_known_args()
SHOW_PREVIEW = _args.preview

# ─── TUNING ───────────────────────────────────────────────────────────────────

# Camera / IMX500
FRAME_W, FRAME_H = 640, 640
TARGET_FPS       = 15
KP_CONF          = 0.10
DETECTION_CONF   = 0.20
MIN_BOX_PX       = 15

# Tracker
IOU_GATE         = 0.05
DIST_GATE        = 200
MAX_MISSING      = 15
REID_FRAMES      = 45
MIN_TRACK_AGE    = 10
KF_K_POS         = 0.55
KF_K_VEL         = 0.25
MIN_VOTES        = 2

# BLE fusion
BLE_POLL_SECS        = 5
BLE_STALE_SECS       = 15
BLE_RSSI_STRONG_MAX  = 55
BLE_RSSI_MEDIUM_MAX  = 75
MIN_PERSIST_FRAMES   = 120

# MiVolo
MIVOLO_MIN_CONF       = 0.45
MIVOLO_REINFER_FRAMES = 60
AGE_BUCKETS = [
    (0,  13,  "child"),
    (13, 31,  "young_adult"),
    (31, 61,  "adult"),
    (61, 120, "senior"),
]

# Preview
PREVIEW_SCALE     = 0.5
COLOUR_EMPLOYEE   = (0, 165, 255)
COLOUR_CUSTOMER_M = (255, 140, 0)
COLOUR_CUSTOMER_F = (0, 200, 120)
COLOUR_UNKNOWN    = (150, 150, 150)
FONT              = cv2.FONT_HERSHEY_SIMPLEX

MIN_TRACK_SECS = 2

# ─── NEW (capture) ─── command poll interval (seconds)
COMMAND_POLL_SECS = 3

# ─── IMX500 / PICAMERA2 IMPORTS ───────────────────────────────────────────────

try:
    from picamera2 import Picamera2, Preview
    from picamera2.devices.imx500 import IMX500
    from picamera2.devices.imx500.postprocess_highernet import postprocess_higherhrnet
    from picamera2 import MappedArray, CompletedRequest
    PICAM_OK = True
except ImportError as _e:
    log.error(f"picamera2/IMX500 not available: {_e}")
    PICAM_OK = False

# ─── MIVOLO IMPORTS ───────────────────────────────────────────────────────────

try:
    import mivolo.predictor
    from mivolo.predictor import Predictor as _MiVoloPredictor
    from mivolo.structures import PersonAndFaceResult
    from ultralytics.engine.results import Results as _ULResults
    import torch
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    MIVOLO_OK = True
except ImportError as _e:
    log.warning(f"MiVolo not available: {_e}. Age/gender will be 'unknown'.")
    MIVOLO_OK = False

# ─── SHARED STATE ─────────────────────────────────────────────────────────────

_shared: dict = {"frame": None, "persons": [], "lock": threading.Lock(), "annotated_frame": None}

# ─── ZONE CACHE ───────────────────────────────────────────────────────────────

class ZoneCache:
    def __init__(self):
        self._zones: list = []
        self._lock = threading.Lock()
        self._load_disk()
        self._fetch()

    def _fetch(self) -> None:
        try:
            r = requests.post(
                f"{EDGE_URL}/reload-zones",
                headers=_DEVICE_HEADERS,
                json={"camera_id": CAMERA_ID},
                timeout=15,
            )
            r.raise_for_status()
            zones = r.json().get("zones", [])
            with self._lock:
                self._zones = zones
            ZONE_CACHE_PATH.write_text(json.dumps(zones))
            log.info(f"Zones refreshed: {len(zones)} zone(s)")
        except Exception as exc:
            log.warning(f"Zone fetch failed (using disk cache): {exc}")

    def _load_disk(self) -> None:
        try:
            self._zones = json.loads(ZONE_CACHE_PATH.read_text())
            log.info(f"Zones loaded from disk: {len(self._zones)} zone(s)")
        except Exception:
            self._zones = []

    def get(self) -> list:
        with self._lock:
            return list(self._zones)

    def refresh_loop(self) -> None:
        while not _shutdown.is_set():
            _shutdown.wait(timeout=1800)
            if not _shutdown.is_set():
                self._fetch()

ZONES = ZoneCache()

# ─── EMPLOYEE CACHE ───────────────────────────────────────────────────────────

_emp_lock       = threading.Lock()
_emp_cache: dict = {}
_emp_flag_mtime = 0.0

def _load_emp_cache() -> dict:
    try:
        rows = json.loads(EMPLOYEES_CACHE_PATH.read_text())
        return {r["employee_id"]: r for r in rows if "employee_id" in r}
    except Exception:
        return {}

def _reload_emp_if_updated() -> None:
    global _emp_cache, _emp_flag_mtime
    try:
        mtime = EMPLOYEES_RELOAD_FLAG.stat().st_mtime if EMPLOYEES_RELOAD_FLAG.exists() else 0.0
        if mtime <= _emp_flag_mtime:
            return
        _emp_flag_mtime = mtime
        new = _load_emp_cache()
        with _emp_lock:
            _emp_cache = new
        log.info(f"Employee cache hot-reloaded: {len(new)} employee(s)")
    except Exception as exc:
        log.warning(f"Employee reload failed: {exc}")

def get_employee(employee_id: str) -> Optional[dict]:
    with _emp_lock:
        return _emp_cache.get(employee_id)

# ─── BLE STATE ────────────────────────────────────────────────────────────────

_ble_lock  = threading.Lock()
_ble_state = {"count": 0, "employees": []}

def _ble_reader_loop() -> None:
    global _ble_state
    while not _shutdown.is_set():
        try:
            if BLE_STATE_PATH.exists():
                raw = json.loads(BLE_STATE_PATH.read_text())
                ts  = datetime.fromisoformat(raw.get("ts", "2000-01-01T00:00:00+00:00"))
                age = (datetime.now(timezone.utc) - ts).total_seconds()
                if age <= BLE_STALE_SECS:
                    with _ble_lock:
                        _ble_state = {
                            "count":     int(raw.get("count", 0)),
                            "employees": raw.get("employees", []),
                        }
                else:
                    with _ble_lock:
                        _ble_state = {"count": 0, "employees": []}
        except Exception:
            pass
        _shutdown.wait(timeout=BLE_POLL_SECS)

def ble_employees() -> list:
    with _ble_lock:
        return list(_ble_state["employees"])

# ─── GEOMETRY ─────────────────────────────────────────────────────────────────

def _body_box(kpts: np.ndarray) -> Optional[list]:
    vis = kpts[kpts[:, 2] > KP_CONF]
    if len(vis) < 2:
        return None
    x1, x2 = float(vis[:, 0].min()), float(vis[:, 0].max())
    y1, y2 = float(vis[:, 1].min()), float(vis[:, 1].max())
    pw = max((x2 - x1) * 0.25, 15)
    ph = max((y2 - y1) * 0.25, 15)
    return [max(0, x1-pw), max(0, y1-ph), min(FRAME_W, x2+pw), min(FRAME_H, y2+ph)]

def _face_box(kpts: np.ndarray) -> Optional[list]:
    KP = dict(nose=0, l_eye=1, r_eye=2, l_ear=3, r_ear=4)
    la, ra = kpts[KP["l_ear"]], kpts[KP["r_ear"]]
    le, re = kpts[KP["l_eye"]], kpts[KP["r_eye"]]
    n      = kpts[KP["nose"]]
    if la[2] > KP_CONF and ra[2] > KP_CONF:
        d = float(np.linalg.norm(la[:2] - ra[:2]))
        if d > 4:
            cx, cy, sz = float((la[0]+ra[0])/2), float((la[1]+ra[1])/2), d*1.4
            return [cx-sz, cy-sz*1.1, cx+sz, cy+sz*0.9]
    if le[2] > KP_CONF and re[2] > KP_CONF:
        d = float(np.linalg.norm(le[:2] - re[:2]))
        if d > 4:
            cx, cy, sz = float((le[0]+re[0])/2), float((le[1]+re[1])/2), d*2.0
            return [cx-sz, cy-sz, cx+sz, cy+sz*1.4]
    if n[2] > KP_CONF:
        return [float(n[0])-30, float(n[1])-30, float(n[0])+30, float(n[1])+30]
    return None

def _iou(a: list, b: list) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-6)

def _cx_cy(box: list) -> tuple:
    return (box[0]+box[2])/2, (box[1]+box[3])/2

def _dist(a: list, b: list) -> float:
    ax, ay = _cx_cy(a); bx, by = _cx_cy(b)
    return ((ax-bx)**2 + (ay-by)**2)**0.5

def _nms(persons: list, thresh: float) -> list:
    if len(persons) <= 1:
        return persons
    persons = sorted(persons, key=lambda p: p["score"], reverse=True)
    keep, supp = [], set()
    for i, p in enumerate(persons):
        if i in supp:
            continue
        keep.append(p)
        for j in range(i+1, len(persons)):
            if j not in supp and _iou(p["box"], persons[j]["box"]) > thresh:
                supp.add(j)
    return keep

def _point_in_polygon(x: float, y: float, pts: list) -> bool:
    n = len(pts); inside = False; j = n - 1
    for i in range(n):
        xi, yi = pts[i][0], pts[i][1]
        xj, yj = pts[j][0], pts[j][1]
        if ((yi > y) != (yj > y)) and (x < (xj-xi)*(y-yi)/(yj-yi+1e-9)+xi):
            inside = not inside
        j = i
    return inside

def _zone_for_point(cx: float, cy: float, zones: list) -> Optional[str]:
    for z in zones:
        pts = z.get("points")
        if pts and len(pts) >= 3:
            if _point_in_polygon(cx, cy, pts):
                return z.get("name")
        else:
            if z.get("x1",0) <= cx <= z.get("x2",FRAME_W) and \
               z.get("y1",0) <= cy <= z.get("y2",FRAME_H):
                return z.get("name")
    return None

def _map_age(age) -> str:
    try:
        a = float(age)
    except (TypeError, ValueError):
        return "unknown"
    for lo, hi, label in AGE_BUCKETS:
        if lo <= a < hi:
            return label
    return "senior"

def _map_gender(gen, conf: float) -> str:
    if gen is None or conf < MIVOLO_MIN_CONF:
        return "unknown"
    g = str(gen).lower()
    if "female" in g:
        return "female"
    if "male" in g:
        return "male"
    return "unknown"

# ─── TRACK ────────────────────────────────────────────────────────────────────

_track_id_lock    = threading.Lock()
_track_id_counter = 0

def _next_track_id() -> int:
    global _track_id_counter
    with _track_id_lock:
        _track_id_counter += 1
        return _track_id_counter

class Track:
    def __init__(self, box: list, kpts: np.ndarray, entered_at: datetime):
        self.track_id    = _next_track_id()
        self.entered_at  = entered_at
        self.kpts        = kpts
        self.missing     = 0
        self.age_frames  = 0
        self.infer_frame = 0

        self.gender        = "unknown"
        self.age_bracket   = "unknown"
        self._gen_votes: dict = defaultdict(int)
        self._age_votes: dict = defaultdict(int)
        self.gender_stable = False
        self.age_stable    = False

        self.is_employee = False
        self.beacon_mac: Optional[str] = None

        self._zone_frames: dict = defaultdict(int)
        self.last_zone: Optional[str] = None

        cx, cy = _cx_cy(box)
        w, h   = box[2]-box[0], box[3]-box[1]
        self._s = np.array([cx, cy, w, h, 0., 0.], dtype=np.float64)

    def predict(self) -> None:
        self._s[0] += self._s[4]
        self._s[1] += self._s[5]

    @property
    def predicted_box(self) -> list:
        cx, cy, w, h = self._s[0], self._s[1], self._s[2], self._s[3]
        return [cx-w/2, cy-h/2, cx+w/2, cy+h/2]

    @property
    def box(self) -> list:
        return self.predicted_box

    def update(self, new_box: list, kpts: np.ndarray, zone: Optional[str]) -> None:
        ncx, ncy = _cx_cy(new_box)
        nw, nh   = new_box[2]-new_box[0], new_box[3]-new_box[1]
        self._s[0] += KF_K_POS * (ncx - self._s[0])
        self._s[1] += KF_K_POS * (ncy - self._s[1])
        self._s[2] += 0.2      * (nw  - self._s[2])
        self._s[3] += 0.2      * (nh  - self._s[3])
        self._s[4] += KF_K_VEL * ((ncx - self._s[0]) - self._s[4])
        self._s[5] += KF_K_VEL * ((ncy - self._s[1]) - self._s[5])
        self.kpts    = kpts
        self.missing = 0
        self.age_frames += 1
        if zone:
            self._zone_frames[zone] += 1
            self.last_zone = zone

    def add_votes(self, age_bracket: str, gender: str) -> None:
        if gender != "unknown":
            self._gen_votes[gender] += 1
            total  = sum(self._gen_votes.values())
            winner = max(self._gen_votes, key=self._gen_votes.get)
            if total >= MIN_VOTES and self._gen_votes[winner] / total > 0.55:
                self.gender        = winner
                self.gender_stable = True
        if age_bracket != "unknown":
            self._age_votes[age_bracket] += 1
            total  = sum(self._age_votes.values())
            winner = max(self._age_votes, key=self._age_votes.get)
            if total >= MIN_VOTES and self._age_votes[winner] / total > 0.55:
                self.age_bracket = winner
                self.age_stable  = True

    def needs_inference(self, current_frame: int) -> bool:
        return current_frame - self.infer_frame >= MIVOLO_REINFER_FRAMES

    def majority_zone(self) -> Optional[str]:
        if not self._zone_frames:
            return self.last_zone
        return max(self._zone_frames, key=self._zone_frames.get)

    def duration_secs(self, exited_at: datetime) -> int:
        return max(0, int((exited_at - self.entered_at).total_seconds()))

    def colour(self) -> tuple:
        if self.is_employee:
            return COLOUR_EMPLOYEE
        if self.gender == "male":
            return COLOUR_CUSTOMER_M
        if self.gender == "female":
            return COLOUR_CUSTOMER_F
        return COLOUR_UNKNOWN

# ─── TRACK STORE ──────────────────────────────────────────────────────────────

class TrackStore:
    def __init__(self):
        self._tracks:   dict = {}
        self._reid_buf: dict = {}

    def update(self, persons: list, now: datetime, zones: list) -> tuple:
        for t in self._tracks.values():
            t.predict()

        track_list = list(self._tracks.values())
        pairs, unmatched_t, unmatched_d = self._match(track_list, persons)

        for ti, di in pairs:
            det  = persons[di]
            zone = _zone_for_point(*_cx_cy(det["box"]), zones)
            track_list[ti].update(det["box"], det["kpts"], zone)

        completed = []
        for ti in unmatched_t:
            t = track_list[ti]
            t.missing += 1
            if t.missing > MAX_MISSING:
                del self._tracks[t.track_id]
                t._exited_at = now
                if t.age_frames >= MIN_TRACK_AGE:
                    completed.append(t)
                self._reid_buf[t.track_id] = [list(t.box), REID_FRAMES, t]

        claimed = set()
        for di in unmatched_d:
            det_box  = persons[di]["box"]
            best_uid, best_d = None, DIST_GATE * 1.5
            for uid, (last_box, ttl, _) in self._reid_buf.items():
                if uid in claimed:
                    continue
                d = _dist(det_box, last_box)
                if d < best_d:
                    best_d, best_uid = d, uid
            if best_uid is not None:
                _, _, old_t = self._reid_buf.pop(best_uid)
                zone = _zone_for_point(*_cx_cy(det_box), zones)
                old_t.update(det_box, persons[di]["kpts"], zone)
                old_t.missing = 0
                self._tracks[old_t.track_id] = old_t
                claimed.add(best_uid)
            else:
                t    = Track(det_box, persons[di]["kpts"], now)
                zone = _zone_for_point(*_cx_cy(det_box), zones)
                t.last_zone = zone
                self._tracks[t.track_id] = t

        for uid in list(self._reid_buf):
            b, ttl, t = self._reid_buf[uid]
            if ttl <= 1:
                del self._reid_buf[uid]
            else:
                self._reid_buf[uid] = [b, ttl-1, t]

        return list(self._tracks.values()), completed

    def _match(self, tracks: list, persons: list) -> tuple:
        if not tracks or not persons:
            return [], list(range(len(tracks))), list(range(len(persons)))
        n, m  = len(tracks), len(persons)
        score = np.zeros((n, m), dtype=np.float32)
        for i, t in enumerate(tracks):
            for j, d in enumerate(persons):
                iou_v = _iou(t.predicted_box, d["box"])
                dv    = _dist(t.predicted_box, d["box"])
                if iou_v >= IOU_GATE or dv <= DIST_GATE:
                    score[i, j] = iou_v * 0.6 + max(0.0, 1.0 - dv / DIST_GATE) * 0.4
        matched_t, matched_d, pairs = set(), set(), []
        for idx in np.argsort(-score.ravel()):
            i, j = divmod(int(idx), m)
            if score[i, j] <= 0:
                break
            if i in matched_t or j in matched_d:
                continue
            pairs.append((i, j))
            matched_t.add(i); matched_d.add(j)
        return pairs, [i for i in range(n) if i not in matched_t], \
                      [j for j in range(m) if j not in matched_d]

    def active(self) -> list:
        return list(self._tracks.values())

    def visible(self) -> list:
        return [t for t in self._tracks.values() if t.missing == 0]

STORE = TrackStore()

# ─── BLE EMPLOYEE EXCLUSION ───────────────────────────────────────────────────

def _rssi_zone(rssi: int) -> str:
    a = abs(rssi)
    if a <= BLE_RSSI_STRONG_MAX:
        return "strong"
    if a <= BLE_RSSI_MEDIUM_MAX:
        return "medium"
    return "weak"

def apply_ble_exclusion(visible_tracks: list) -> None:
    beacons = ble_employees()
    if not beacons:
        for t in visible_tracks:
            t.is_employee = False
        return

    beacons_sorted = sorted(beacons, key=lambda b: abs(b.get("rssi", -99)))
    claimed: set = set()

    for beacon in beacons_sorted:
        rssi     = beacon.get("rssi", -99)
        zone     = _rssi_zone(rssi)
        emp_id   = beacon.get("employee_id", "")
        emp_info = get_employee(emp_id)
        emp_gen  = emp_info["gender"] if emp_info else "unknown"

        if zone == "weak":
            continue

        candidates = []
        for t in visible_tracks:
            if id(t) in claimed:
                continue
            if emp_gen != "unknown" and t.gender not in ("unknown", emp_gen):
                continue
            if zone == "medium" and t.age_frames < MIN_PERSIST_FRAMES:
                continue
            candidates.append(t)

        if not candidates:
            continue

        best = max(candidates, key=lambda t: t.age_frames)
        if not best.is_employee:
            log.info(
                f"Track #{best.track_id} → STAFF "
                f"(emp={emp_id[:8]} rssi={rssi} zone={zone} frames={best.age_frames})"
            )
        best.is_employee = True
        best.beacon_mac  = (emp_info or {}).get("beacon_mac") or beacon.get("beacon_mac")
        claimed.add(id(best))

    for t in visible_tracks:
        if t.is_employee and id(t) not in claimed:
            log.info(f"Track #{t.track_id} employee flag cleared (beacon gone)")
            t.is_employee = False
            t.beacon_mac  = None

# ─── SQLITE BUFFER ────────────────────────────────────────────────────────────

_db_write_queue: queue.Queue = queue.Queue()

def _db_writer_loop() -> None:
    con = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("""
        CREATE TABLE IF NOT EXISTS pending_tracks (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            payload    TEXT    NOT NULL,
            created_at TEXT    NOT NULL
        )
    """)
    con.commit()
    log.info(f"SQLite WAL writer ready: {DB_PATH}")

    while not _shutdown.is_set():
        try:
            item = _db_write_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        if item is None:
            # Sentinel: drain any remaining items already in the queue before stopping.
            # task_done() for the sentinel itself so Queue.join() can unblock.
            _db_write_queue.task_done()
            break
        payload_json, ts = item
        try:
            con.execute(
                "INSERT INTO pending_tracks (payload, created_at) VALUES (?, ?)",
                (payload_json, ts)
            )
            con.commit()
        except Exception as exc:
            log.error(f"SQLite write failed: {exc}")
        finally:
            # Always mark done so Queue.join() in shutdown works correctly.
            _db_write_queue.task_done()

    con.close()

def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def store_track(track_dict: dict) -> None:
    _db_write_queue.put((json.dumps(track_dict), datetime.utcnow().isoformat()))

def flush_tracks() -> None:
    """
    Upload pending tracks to Supabase via upload-tracks Edge Function.
    Reads+closes SQLite before network call to avoid holding a lock during I/O.
    Only deletes rows from SQLite after a confirmed successful upload.
    """
    try:
        con  = sqlite3.connect(str(DB_PATH), timeout=5)
        con.execute("PRAGMA journal_mode=WAL")
        rows = con.execute("SELECT id, payload FROM pending_tracks ORDER BY id").fetchall()
        con.close()
    except Exception as exc:
        log.error(f"flush_tracks: SQLite read failed: {exc}")
        return

    if not rows:
        log.info("Upload: no pending tracks")
        return

    ids    = [r[0] for r in rows]
    tracks = [json.loads(r[1]) for r in rows]
    log.info(
        f"Upload: flushing {len(tracks)} tracks "
        f"[org={ORG_ID} store={STORE_ID} camera={CAMERA_ID}]"
    )

    BATCH = 500
    total = 0
    uploaded_ids: list = []

    try:
        for i in range(0, len(tracks), BATCH):
            batch     = tracks[i:i+BATCH]
            batch_ids = ids[i:i+BATCH]
            resp = requests.post(
                f"{EDGE_URL}/upload-tracks",
                headers=_DEVICE_HEADERS,
                json={"device_id": CAMERA_ID, "tracks": batch},
                timeout=30,
            )
            resp.raise_for_status()
            n = resp.json().get("inserted", 0)
            total += n
            uploaded_ids.extend(batch_ids)
            log.info(f"Upload: batch {i//BATCH+1} → {n} inserted")

    except Exception as exc:
        log.error(f"Upload failed (tracks kept in buffer for next attempt): {exc}")
        # If some batches succeeded before the failure, clear only those rows.
        if not uploaded_ids:
            return

    if not uploaded_ids:
        return

    try:
        con = sqlite3.connect(str(DB_PATH), timeout=5)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute(
            f"DELETE FROM pending_tracks WHERE id IN ({','.join('?'*len(uploaded_ids))})",
            uploaded_ids,
        )
        con.commit()
        con.close()
        log.info(f"Upload: complete — {total} tracks inserted, {len(uploaded_ids)} rows cleared")
    except Exception as exc:
        log.error(f"flush_tracks: SQLite delete failed (duplicate upload possible next run): {exc}")

def buffer_size() -> int:
    try:
        con = sqlite3.connect(str(DB_PATH), timeout=1)
        n   = con.execute("SELECT COUNT(*) FROM pending_tracks").fetchone()[0]
        con.close()
        return n
    except Exception:
        return 0

# ─── HEARTBEAT ────────────────────────────────────────────────────────────────

def send_heartbeat(device_ok: bool = True, reason: str = "") -> None:
    tag = f" ({reason})" if reason else ""
    try:
        r = requests.post(
            f"{EDGE_URL}/process-heartbeat",
            headers=_DEVICE_HEADERS,
            json={
                "camera_id":        CAMERA_ID,
                "device_ok":        device_ok,
                "firmware_version": "2.5.0",
            },
            timeout=10,
        )
        r.raise_for_status()
        log.info(f"Heartbeat: ok={device_ok}{tag}")
    except Exception as exc:
        log.warning(f"Heartbeat failed (will retry in 15 min): {exc}")

# ─── NEW (capture) ─── COMMAND POLL + CAPTURE HANDLER ─────────────────────────

def _handle_capture(command_id: str) -> None:
    """
    Grab the latest frame and POST it to the `capture-snapshot` edge function with
    {camera_id, token, command_id}. The function verifies the device, uploads to the
    snapshots bucket, and closes the device_commands row (status=done, image_url) so
    the app can preview it. Uses the annotated frame; swap to _shared["frame"] for a
    clean (unboxed) still.
    """
    with _shared["lock"]:
        jpg = _shared.get("annotated_frame")
    if jpg is None:
        log.warning("capture: no frame available yet")
        return
    try:
        r = requests.post(
            f"{EDGE_URL}/capture-snapshot",
            headers=_DEVICE_HEADERS,
            json={
                "camera_id":    CAMERA_ID,
                "token":        DEVICE_TOKEN,   # raw; function hashes + verifies vs devices
                "section_name": "entry",
                "image_base64": base64.b64encode(jpg).decode("ascii"),
                "command_id":   command_id,
            },
            timeout=30,
        )
        r.raise_for_status()
        log.info(f"snapshot uploaded for command {command_id}")
    except Exception as exc:
        log.error(f"snapshot upload failed: {exc}")

def _command_poll_loop() -> None:
    """
    Poll the `poll-commands` edge function for pending 'capture' commands targeted
    at this device's store, and handle each. Runs until shutdown.
    """
    while not _shutdown.is_set():
        try:
            r = requests.post(
                f"{EDGE_URL}/poll-commands",
                headers=_DEVICE_HEADERS,
                json={"camera_id": CAMERA_ID, "token": DEVICE_TOKEN},
                timeout=10,
            )
            r.raise_for_status()
            for cmd in r.json().get("commands", []):
                if cmd.get("type") == "capture":
                    _handle_capture(cmd["id"])
        except Exception as exc:
            log.debug(f"command poll failed: {exc}")
        _shutdown.wait(timeout=COMMAND_POLL_SECS)

# ─── SCHEDULER ────────────────────────────────────────────────────────────────

def _setup_scheduler() -> None:
    for t in UPLOAD_TIMES_UTC:
        schedule.every().day.at(t).do(flush_tracks)
    schedule.every(15).minutes.do(send_heartbeat)
    log.info(f"Scheduler: uploads at {UPLOAD_TIMES_UTC} UTC | heartbeat every 15 min")

def _scheduler_loop() -> None:
    while not _shutdown.is_set():
        schedule.run_pending()
        _shutdown.wait(timeout=30)

# ─── CAMERA PRE-CALLBACK ──────────────────────────────────────────────────────

def _pre_callback(request, imx500_ref) -> None:
    """
    Runs in the picamera2 camera thread.
    Parses IMX500 pose metadata and copies frame into shared state.
    DEBUG logging shows raw detection scores to diagnose threshold issues.
    """
    np_outputs = imx500_ref.get_outputs(metadata=request.get_metadata(), add_batch=True)
    if np_outputs is None:
        log.debug("_pre_callback: np_outputs is None — no inference output yet")
        return

    log.debug(f"_pre_callback: np_outputs received, {len(np_outputs)} tensor(s)")

    try:
        keypoints_raw, scores, _ = postprocess_higherhrnet(
            outputs=np_outputs,
            img_size=(FRAME_H, FRAME_W),
            img_w_pad=(0, 0), img_h_pad=(0, 0),
            detection_threshold=KP_CONF,
            network_postprocess=True,
        )
    except Exception as exc:
        log.warning(f"postprocess_higherhrnet failed: {exc}")
        return

    with MappedArray(request, "main") as m:
        frame = m.array.copy()

    persons = []
    if scores is not None and len(scores) > 0:
        log.debug(
            f"_pre_callback: {len(scores)} raw detection(s), "
            f"scores={[round(float(s),3) for s in scores]}"
        )
        kpts_all = np.reshape(np.stack(keypoints_raw, axis=0), (len(scores), 17, 3))
        for kpts, score in zip(kpts_all, scores):
            if score < DETECTION_CONF:
                log.debug(f"_pre_callback: score {score:.3f} < DETECTION_CONF {DETECTION_CONF} — skipped")
                continue
            b = _body_box(kpts)
            if b is None:
                log.debug("_pre_callback: body_box returned None — too few confident keypoints")
                continue
            if (b[2]-b[0]) < MIN_BOX_PX or (b[3]-b[1]) < MIN_BOX_PX:
                log.debug(f"_pre_callback: box too small ({b[2]-b[0]:.0f}x{b[3]-b[1]:.0f}px)")
                continue
            persons.append({"box": b, "kpts": kpts, "score": float(score)})
    else:
        log.debug("_pre_callback: no raw detections this frame")

    persons = _nms(persons, 0.45)
    if persons:
        log.info(
            f"_pre_callback: {len(persons)} person(s) after NMS | "
            f"scores={[round(p['score'],3) for p in persons]}"
        )

    with _shared["lock"]:
        _shared["frame"]   = frame
        _shared["persons"] = persons

# ─── MIVOLO WORKER ────────────────────────────────────────────────────────────

_mivolo_result_q: queue.Queue = queue.Queue()
_mivolo_infer_q:  queue.Queue = queue.Queue(maxsize=8)
_mivolo_executor = ThreadPoolExecutor(max_workers=2)

def _mivolo_worker_loop() -> None:
    if not MIVOLO_OK:
        return

    class _NoOpDetector:
        def __init__(self, *a, **k): pass

    mivolo.predictor.Detector = _NoOpDetector

    from types import SimpleNamespace
    cfg = SimpleNamespace(
        checkpoint=MIVOLO_PATH, detector_weights=YOLO_PATH,
        device="cpu", with_persons=True, disable_faces=False,
        draw=False, resize=224,
    )
    try:
        predictor = _MiVoloPredictor(cfg, verbose=False)
        log.info(f"MiVolo ready: {MIVOLO_PATH}")
    except Exception as exc:
        log.error(f"MiVolo init failed: {exc}")
        return

    while not _shutdown.is_set():
        try:
            item = _mivolo_infer_q.get(timeout=0.5)
        except queue.Empty:
            continue

        track_id, frame, body_b, face_b = item
        try:
            tensors = [[body_b[0], body_b[1], body_b[2], body_b[3], 0.9, 0.0]]
            if face_b:
                tensors.append([face_b[0], face_b[1], face_b[2], face_b[3], 0.9, 1.0])
            else:
                tensors.append([body_b[0], body_b[1], body_b[2], body_b[3], 0.5, 1.0])

            t_boxes = torch.tensor(tensors, dtype=torch.float32)
            res     = _ULResults(
                orig_img=frame, path="",
                names={0: "person", 1: "face"}, boxes=t_boxes
            )
            pr = PersonAndFaceResult(res)
            predictor.age_gender_model.predict(frame, pr)

            for i, idx in enumerate(pr.get_bboxes_inds("person")):
                age_raw = (float(pr.ages[idx])
                           if pr.ages is not None and idx < len(pr.ages)
                              and pr.ages[idx] is not None else 25.0)
                gen_raw = (str(pr.genders[idx]).lower()
                           if pr.genders is not None and idx < len(pr.genders)
                              and pr.genders[idx] is not None else "")
                gconf   = (float(pr.gender_scores[idx])
                           if hasattr(pr, "gender_scores") and pr.gender_scores is not None
                              and idx < len(pr.gender_scores) else 0.5)
                age_bracket = _map_age(age_raw)
                gender      = _map_gender(gen_raw, gconf)
                _mivolo_result_q.put((track_id, age_bracket, gender))
                break
        except Exception as exc:
            log.debug(f"MiVolo infer error: {exc}")

def _submit_mivolo(track: Track, frame: np.ndarray, current_frame: int) -> None:
    if not MIVOLO_OK or not track.needs_inference(current_frame):
        return
    b    = track.box
    face = _face_box(track.kpts)
    track.infer_frame = current_frame
    try:
        _mivolo_infer_q.put_nowait((track.track_id, frame, b, face))
    except queue.Full:
        pass

def _drain_mivolo_results() -> None:
    track_map = {t.track_id: t for t in STORE.active()}
    while True:
        try:
            tid, age_bracket, gender = _mivolo_result_q.get_nowait()
            t = track_map.get(tid)
            if t:
                t.add_votes(age_bracket, gender)
        except queue.Empty:
            break

# ─── WEB PREVIEW SERVER ───────────────────────────────────────────────────────

_WEB_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Vidar Preview — {camera_id}</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d0d0d; color: #e0e0e0; font-family: 'Courier New', monospace;
         display: flex; flex-direction: column; align-items: center; min-height: 100vh; }}
  header {{ width: 100%; background: #161616; border-bottom: 1px solid #2a2a2a;
            padding: 10px 20px; display: flex; justify-content: space-between; align-items: center; }}
  header h1 {{ font-size: 14px; letter-spacing: 2px; color: #00e5a0; text-transform: uppercase; }}
  header span {{ font-size: 12px; color: #555; }}
  #feed {{ width: 100%; max-width: 900px; margin: 20px auto 0; padding: 0 16px; }}
  #feed img {{ width: 100%; border: 1px solid #222; border-radius: 4px; display: block; }}
  #stats {{ width: 100%; max-width: 900px; margin: 16px auto 20px; padding: 0 16px;
            display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; }}
  .card {{ background: #161616; border: 1px solid #222; border-radius: 4px;
           padding: 12px 16px; }}
  .card .label {{ font-size: 10px; color: #555; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 4px; }}
  .card .value {{ font-size: 26px; font-weight: bold; color: #00e5a0; }}
  .card .value.warn {{ color: #ffaa00; }}
  .card .value.dim  {{ color: #666; font-size: 14px; margin-top: 4px; }}
  #log {{ width: 100%; max-width: 900px; margin: 0 auto 30px; padding: 0 16px; }}
  #log pre {{ background: #111; border: 1px solid #1e1e1e; border-radius: 4px;
              padding: 10px 14px; font-size: 11px; color: #555; max-height: 120px;
              overflow-y: auto; white-space: pre-wrap; }}
</style>
</head>
<body>
<header>
  <h1>&#9673; Vidar Live &nbsp;|&nbsp; {camera_id}</h1>
  <span id="ts">connecting…</span>
</header>
<div id="feed">
  <img src="/stream" alt="live feed">
</div>
<div id="stats">
  <div class="card"><div class="label">Customers</div><div class="value" id="s-cust">—</div></div>
  <div class="card"><div class="label">Staff (cam)</div><div class="value" id="s-staff">—</div></div>
  <div class="card"><div class="label">Staff (BLE)</div><div class="value" id="s-ble">—</div></div>
  <div class="card"><div class="label">Active tracks</div><div class="value" id="s-active">—</div></div>
  <div class="card"><div class="label">Buffer</div><div class="value" id="s-buf">—</div></div>
  <div class="card"><div class="label">IST</div><div class="value dim" id="s-ist">—</div></div>
</div>
<div id="log"><pre id="logbox">waiting for data…</pre></div>
<script>
async function poll() {{
  try {{
    const r = await fetch('/stats');
    const d = await r.json();
    document.getElementById('s-cust').textContent   = d.customers;
    document.getElementById('s-staff').textContent  = d.staff_cam;
    document.getElementById('s-ble').textContent    = d.staff_ble;
    document.getElementById('s-active').textContent = d.active_tracks;
    const buf = d.buffer_size;
    const el  = document.getElementById('s-buf');
    el.textContent = buf;
    el.className   = 'value' + (buf > 500 ? ' warn' : '');
    document.getElementById('s-ist').textContent = d.ist;
    document.getElementById('ts').textContent    = 'updated ' + new Date().toLocaleTimeString();
    const box = document.getElementById('logbox');
    box.textContent = (d.recent_tracks || []).map(t =>
      `[${{t.entered_at_ist}}] #${{t.track_id}} ${{t.role}} ${{t.gender}}/${{t.age_bracket}} zone=${{t.zone || '—'}} dwell=${{t.dwell}}s`
    ).join('\\n') || 'no recent tracks';
    box.scrollTop = box.scrollHeight;
  }} catch(e) {{ document.getElementById('ts').textContent = 'offline'; }}
}}
poll();
setInterval(poll, 2000);
</script>
</body>
</html>"""

# Ring buffer of recently completed tracks for the web UI log
_recent_tracks: deque = deque(maxlen=30)
_recent_tracks_lock   = threading.Lock()

def _record_recent_track(payload: dict, track_id: int) -> None:
    """Store a summary of a completed track for the web preview log."""
    try:
        entered_ist = datetime.fromisoformat(payload["entered_at"]) \
                              .astimezone(IST).strftime("%H:%M:%S")
    except Exception:
        entered_ist = "?"
    with _recent_tracks_lock:
        _recent_tracks.append({
            "track_id":      track_id,
            "role":          "STAFF" if payload["is_employee"] else "CUST",
            "gender":        payload.get("gender", "unknown"),
            "age_bracket":   payload.get("age_bracket", "unknown"),
            "zone":          payload.get("zone_name") or "",
            "dwell":         payload.get("dwell_seconds", 0),
            "entered_at_ist": entered_ist,
        })


def _start_web_preview() -> None:
    """Start Flask MJPEG + stats server in a daemon thread."""
    if not FLASK_OK:
        log.warning("Flask not installed — web preview unavailable. pip install flask")
        return

    app = Flask(__name__)
    log_fn = logging.getLogger("werkzeug")
    log_fn.setLevel(logging.ERROR)  # silence Flask request logs

    @app.route("/")
    def index():
        return _WEB_HTML.replace("{camera_id}", CAMERA_ID)

    def _gen_frames():
        while not _shutdown.is_set():
            with _shared["lock"]:
                jpg = _shared.get("annotated_frame")
            if jpg is None:
                time.sleep(0.05)
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(1 / TARGET_FPS)

    @app.route("/stream")
    def stream():
        return Response(_gen_frames(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/stats")
    def stats():
        active  = STORE.active()
        visible = [t for t in active if t.missing == 0]
        with _recent_tracks_lock:
            recent = list(_recent_tracks)
        return jsonify({
            "customers":     sum(1 for t in visible if not t.is_employee),
            "staff_cam":     sum(1 for t in visible if t.is_employee),
            "staff_ble":     len(ble_employees()),
            "active_tracks": len(active),
            "buffer_size":   buffer_size(),
            "ist":           datetime.now(IST).strftime("%H:%M:%S"),
            "recent_tracks": list(reversed(recent)),
        })

    def _run():
        app.run(host="0.0.0.0", port=PREVIEW_PORT, threaded=True, use_reloader=False)

    t = threading.Thread(target=_run, daemon=True, name="web-preview")
    t.start()
    log.info(f"Web preview started → http://0.0.0.0:{PREVIEW_PORT}  (open in browser on same network)")


# ─── PREVIEW / DRAW ───────────────────────────────────────────────────────────

def _draw(frame: np.ndarray, tracks: list, zones: list) -> np.ndarray:
    vis  = frame.copy()
    h, w = vis.shape[:2]

    for z in zones:
        pts = z.get("points")
        if pts and len(pts) >= 3:
            poly = np.array([[int(p[0]), int(p[1])] for p in pts], dtype=np.int32)
            cv2.polylines(vis, [poly], True, (0, 255, 200), 1)
            cx = int(np.mean([p[0] for p in pts]))
            cy = int(np.mean([p[1] for p in pts]))
            cv2.putText(vis, z.get("name", ""), (cx, cy), FONT, 0.4, (0, 255, 200), 1)
        else:
            x1, y1 = int(z.get("x1",0)), int(z.get("y1",0))
            x2, y2 = int(z.get("x2",w)), int(z.get("y2",h))
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,200), 1)

    for t in tracks:
        if t.missing > 0:
            continue
        b  = t.box
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        col = t.colour()
        cv2.rectangle(vis, (x1,y1), (x2,y2), col, 2)
        parts = [f"#{t.track_id}"]
        if t.is_employee:
            parts.append("STAFF")
        else:
            if t.gender      != "unknown": parts.append(t.gender[0].upper())
            if t.age_bracket != "unknown": parts.append(t.age_bracket)
        if t.last_zone: parts.append(f"[{t.last_zone}]")
        label = " ".join(parts)
        cv2.putText(vis, label, (x1, max(y1-6, 10)), FONT, 0.45, col, 1, cv2.LINE_AA)

    n_cust = sum(1 for t in tracks if not t.is_employee and t.missing == 0)
    n_emp  = sum(1 for t in tracks if t.is_employee     and t.missing == 0)
    ist    = datetime.now(IST).strftime("%H:%M:%S")
    ble    = len(ble_employees())
    cv2.rectangle(vis, (0, 0), (w, 26), (20,20,20), -1)
    cv2.putText(vis,
        f"{ist} IST | Cust:{n_cust} Staff:{n_emp}cam/{ble}ble | "
        f"Buffer:{buffer_size()} | cam={CAMERA_ID}",
        (6, 18), FONT, 0.35, (255,255,255), 1, cv2.LINE_AA)

    # Encode for MJPEG web stream (always, regardless of SHOW_PREVIEW)
    ok, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if ok:
        with _shared["lock"]:
            _shared["annotated_frame"] = jpg.tobytes()

    return vis

# ─── TRACK PAYLOAD ────────────────────────────────────────────────────────────

def _build_payload(t: Track, exited_at: datetime) -> Optional[dict]:
    """
    Build the dict that maps 1-to-1 onto the tracks table columns.
    store_id and org_id are required NOT NULL columns — must be included.
    """
    dur = t.duration_secs(exited_at)
    if dur < MIN_TRACK_SECS:
        return None
    return {
        "camera_id":     CAMERA_ID,
        "store_id":      STORE_ID,      # NOT NULL in schema
        "org_id":        ORG_ID,        # NOT NULL in schema
        "beacon_mac":    t.beacon_mac,
        "is_employee":   t.is_employee,
        "zone_name":     t.majority_zone(),
        "section_name":  None,
        "age_bracket":   t.age_bracket,
        "gender":        t.gender,
        "entered_at":    t.entered_at.isoformat(),
        "exited_at":     exited_at.isoformat(),
        "dwell_seconds": dur,
    }

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if not PICAM_OK:
        log.error("picamera2/IMX500 not available — cannot start.")
        return

    log.info(
        f"Vidar detector v2.6 | camera={CAMERA_ID} | store={STORE_ID} | org={ORG_ID} | "
        f"preview={'YES' if SHOW_PREVIEW else 'NO (headless)'} | "
        f"MiVolo={'YES' if MIVOLO_OK else 'NO'} | "
        f"DETECTION_CONF={DETECTION_CONF} KP_CONF={KP_CONF}"
    )

    init_db()

    with _emp_lock:
        global _emp_cache
        _emp_cache = _load_emp_cache()
    log.info(f"Employee cache: {len(_emp_cache)} employee(s)")

    db_writer_thread = threading.Thread(target=_db_writer_loop, daemon=True, name="db-writer")
    db_writer_thread.start()

    threading.Thread(target=_ble_reader_loop,     daemon=True, name="ble-reader").start()
    threading.Thread(target=ZONES.refresh_loop,   daemon=True, name="zone-refresh").start()
    threading.Thread(target=_mivolo_worker_loop,  daemon=True, name="mivolo").start()

    _setup_scheduler()
    threading.Thread(target=_scheduler_loop, daemon=True, name="scheduler").start()

    # ─── NEW (capture) ─── start the capture command poll loop
    threading.Thread(target=_command_poll_loop, daemon=True, name="cmd-poll").start()

    _start_web_preview()

    threading.Thread(
        target=send_heartbeat,
        kwargs={"device_ok": True, "reason": "startup"},
        daemon=True,
    ).start()

    # ── Camera setup ──────────────────────────────────────────────────────────
    imx500 = IMX500(IMX500_MODEL)
    picam2 = Picamera2(imx500.camera_num)

    if SHOW_PREVIEW:
        picam2.start_preview(Preview.QTGL)
        cam_cfg = picam2.create_preview_configuration(
            main={"size": (FRAME_W, FRAME_H), "format": "RGB888"},
            controls={"FrameRate": TARGET_FPS},
            buffer_count=12,
        )
    else:
        cam_cfg = picam2.create_video_configuration(
            main={"size": (FRAME_W, FRAME_H), "format": "RGB888"},
            controls={"FrameRate": TARGET_FPS},
            buffer_count=4,
        )

    picam2.pre_callback = lambda req: _pre_callback(req, imx500)
    imx500.show_network_fw_progress_bar()
    picam2.start(cam_cfg)

    if SHOW_PREVIEW:
        cv2.namedWindow("Vidar", cv2.WINDOW_NORMAL)

    log.info(
        "Camera running. Ctrl+C or SIGTERM to stop. "
        + ("Keys: Q=quit U=upload R=reload-zones" if SHOW_PREVIEW else "")
    )

    frame_count = 0
    last_log    = time.monotonic()

    try:
        while not _shutdown.is_set():
            with _shared["lock"]:
                if _shared["frame"] is None:
                    time.sleep(0.01)
                    continue
                frame   = _shared["frame"].copy()
                persons = list(_shared["persons"])

            now   = datetime.now(timezone.utc)
            zones = ZONES.get()

            active_tracks, completed_tracks = STORE.update(persons, now, zones)
            visible = STORE.visible()

            _drain_mivolo_results()
            _reload_emp_if_updated()
            apply_ble_exclusion(visible)

            for t in visible:
                _submit_mivolo(t, frame, frame_count)

            for t in completed_tracks:
                exited_at = getattr(t, "_exited_at", now)
                payload   = _build_payload(t, exited_at)
                if payload:
                    store_track(payload)
                    _record_recent_track(payload, t.track_id)
                    log.debug(
                        f"Track #{t.track_id} closed | "
                        f"{'STAFF' if payload['is_employee'] else payload['gender']+'/'+payload['age_bracket']} | "
                        f"zone={payload['zone_name']} | dwell={payload['dwell_seconds']}s"
                    )

            # Always draw (updates web MJPEG stream even when headless)
            vis = _draw(frame, active_tracks, zones)

            if SHOW_PREVIEW:
                vis_small = cv2.resize(vis, None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE)
                cv2.imshow("Vidar", vis_small)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("u"):
                    log.info("Manual upload triggered")
                    threading.Thread(target=flush_tracks, daemon=True).start()
                elif key == ord("r"):
                    log.info("Manual zone reload triggered")
                    threading.Thread(target=ZONES._fetch, daemon=True).start()
            else:
                time.sleep(0.001)

            frame_count += 1
            now_mono = time.monotonic()
            if now_mono - last_log >= 60:
                log.info(
                    f"Active:{len(active_tracks)} | "
                    f"Visible:{len(visible)} | "
                    f"Staff:{sum(1 for t in visible if t.is_employee)} | "
                    f"Buffer:{buffer_size()}"
                )
                last_log = now_mono

    except KeyboardInterrupt:
        _shutdown.set()

    finally:
        log.info("Detector stopping — flushing remaining tracks to SQLite...")
        now = datetime.now(timezone.utc)
        for t in STORE.active():
            payload = _build_payload(t, now)
            if payload:
                store_track(payload)

        # Send sentinel to stop the db writer, then wait for it to finish
        # writing every queued item before we call flush_tracks().
        _db_write_queue.put(None)
        log.info("Waiting for SQLite writer to drain...")
        _db_write_queue.join()
        log.info("SQLite writer drained — uploading to Supabase...")

        flush_tracks()

        picam2.stop()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()
        _mivolo_executor.shutdown(wait=False)
        log.info("Detector stopped.")


if __name__ == "__main__":
    main()


