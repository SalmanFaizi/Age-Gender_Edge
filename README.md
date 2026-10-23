<<<<<<< HEAD
# Edge People-Analytics Pipeline: Technical README

**Audience:** Data Analyst team
**Scope:** On-device detection, tracking, age/gender estimation, zone assignment, staff exclusion, and the per-visit data this pipeline produces.
**Hardware:** Raspberry Pi 5 + Sony IMX500 AI Camera
**Main script:** `detector.py` v2.6 (staff beacon scanner: `blescanner.py` v5.3)
**Timestamps:** stored in **UTC** (ISO-8601). Time-of-day logic uses **IST (UTC+05:30)**.

---

## Contents

1. [Overview](#1-overview)
2. [Processing pipeline](#2-processing-pipeline)
3. [Models](#3-models)
4. [Detection](#4-detection)
5. [Tracking](#5-tracking)
6. [Age & gender estimation](#6-age--gender-estimation)
7. [Zone assignment](#7-zone-assignment)
8. [Staff exclusion (BLE fusion)](#8-staff-exclusion-ble-fusion)
9. [Output data](#9-output-data)
10. [Inference from test-run output](#10-inference-from-test-run-output)
11. [Data-quality caveats & biases](#11-data-quality-caveats--biases)
12. [Analysis recipes](#12-analysis-recipes)
13. [Parameters reference](#13-parameters-reference)
14. [Setup & running](#14-setup--running)
15. [Glossary](#15-glossary)

---

## 1. Overview

The pipeline counts people in a camera's view and records one row per visit with:

- **Entry and exit time**, and **dwell time** in seconds
- **Estimated gender**: `male` / `female` / `unknown`
- **Estimated age bracket**: `child` / `young_adult` / `adult` / `senior` / `unknown`
- **Zone**: the user-defined area of the frame where the person spent most time
- **Staff flag**: whether the person was matched to a staff BLE beacon

All vision inference runs on the device. No video, images or face data are stored. Only anonymous per-visit records are kept.

---

## 2. Processing pipeline

```
IMX500 sensor (640×640 @ 15 fps)
   │  HigherHRNet pose model runs ON the camera chip
   ▼
17 keypoints + score per person
   │  filter (score ≥ 0.20) → body box from keypoints → NMS
   ▼
Tracker (IoU + distance matching, constant-velocity smoothing, short re-ID)
   │                                   ▲
   ├──► MiVolo age/gender (Pi CPU) ────┘ votes per track
   ├──► Zone lookup (point-in-polygon / rectangle)
   └──► Staff exclusion ◄── ble_state.json ◄── blescanner.py (iBeacon scan)
   ▼
Track closes → record built → SQLite buffer (local SQLite file)
   ▼
Batch upload 3×/day (03:30, 08:30, 14:30 UTC) → rows deleted locally only after a confirmed upload
```

Supporting threads: SQLite writer, BLE state reader (every 5 s), zone refresh (every 30 min), MiVolo worker, scheduler, and a local web preview (port 8080).

---

## 3. Models

| Model | Runs on | Input | Output |
|---|---|---|---|
| **HigherHRNet (COCO)**: `imx500_network_higherhrnet_coco.rpk` | IMX500 AI accelerator | Full frame 640×640 | 17 keypoints (x, y, conf) + person score, for every person |
| **MiVOLO** `mivolo_d1_224`: checkpoint `model_imdb_cross_person_4.22_99.46.pth.tar` (27.4 M params) | Pi 5 CPU (2 threads) | Body crop + face crop, 224 px | Age (years, continuous) + gender + gender confidence |

- The checkpoint name refers to the published benchmark: IMDB-clean, person+face, **age MAE ≈ 4.22 years, gender accuracy ≈ 99.46%**. These are lab figures on curated images. Expect lower accuracy from overhead and low-resolution in-store views.
- MiVolo's bundled YOLO detector is **disabled**. Person and face boxes come from the pose keypoints instead.

---

## 4. Detection

1. Raw pose output is post-processed with keypoint threshold `KP_CONF = 0.10`.
2. A person is kept only if **score ≥ `DETECTION_CONF` (0.20)**.
3. **Body box** = min/max of keypoints with conf > 0.10, padded 25% on each side (minimum 15 px). Needs ≥ 2 confident keypoints.
4. Boxes smaller than 15 px in width or height are dropped.
5. **Non-maximum suppression** at IoU 0.45 removes duplicate detections.
6. **Face box** is derived from keypoints in this priority: both ears → both eyes → nose (fixed 60 px box) → none.

---

## 5. Tracking

Each frame, existing tracks are predicted forward and matched to new detections.

- **Match score** = 0.6 × IoU + 0.4 × (1 − distance / 200 px). A pair is eligible if IoU ≥ 0.05 or centre distance ≤ 200 px. Greedy assignment, highest score first.
- **Smoothing:** position gain 0.55, velocity gain 0.25, size gain 0.2 (a lightweight Kalman-style filter).
- **Close:** a track unseen for **> 15 frames (~1 s)** is closed.
- **Re-ID:** for **45 frames (~3 s)** after closing, an unmatched detection within 300 px of the last position revives the track.
- **Keep rules:** a closed track becomes a record only if it was matched in **≥ 10 frames** and lasted **≥ 2 s**.

**Key consequence:** one record = one continuous appearance in one camera's view, **not a unique person** (see §11.1).

---

## 6. Age & gender estimation

- Every visible track is submitted to MiVolo at most once per **60 frames (~4 s)**.
- Inference queue size is **8**. If it's full, the request is **dropped**, not queued.
- Each result is a **vote** on the track:
  - Gender is mapped to `unknown` if confidence < **0.45**.
  - Age is mapped to a bracket:

    | Bracket | Age range (years) |
    |---|---|
    | `child` | 0 – 12 |
    | `young_adult` | 13 – 30 |
    | `adult` | 31 – 60 |
    | `senior` | 61 + |

- A label is **committed** once there are **≥ 2 votes** and the leading label has **> 55%** of them. Before that, it stays `unknown`. The label can still change as more votes arrive. The final value is whatever is committed when the track closes.

---

## 7. Zone assignment

- Zones are defined in **frame pixel coordinates (640×640)** as polygons (`points: [[x,y],…]`, ≥ 3 points) or rectangles (`x1, y1, x2, y2`).
- Each matched frame, the track's **box centre** is tested against the zones (first match wins).
- The record's `zone_name` is the **zone with the most frames** for that track (majority vote). It's `null` if the track was never inside a zone, or if no zones are configured.
- Zones are cached locally in `zone_cache.json`, so detection continues if the zone source is unreachable.

---

## 8. Staff exclusion (BLE fusion)

`blescanner.py` scans Bluetooth Low Energy continuously and only acts on **iBeacon** packets whose MAC address matches a registered staff tag. Every other device is ignored.

**Scanner → detector hand-off** (`ble_state.json`, rewritten every 2 s and ignored if > 15 s old):

```json
{"count": 1, "employees": [{"employee_id": "...", "name": "...", "beacon_mac": "...", "rssi": -48}], "ts": "..."}
```

A staff member counts as present until no ping has been heard for **300 s**.

**Matching logic in the detector:**

| RSSI (abs dBm) | Class | Rule |
|---|---|---|
| ≤ 55 | strong | Eligible immediately |
| 56 – 75 | medium | Track must be ≥ 120 frames (~8 s) old |
| > 75 | weak | Ignored |

For each beacon (strongest first), the **longest-lived visible track** is marked as staff, provided its gender doesn't contradict the staff member's recorded gender. The flag is removed if the beacon disappears.

**Hourly BLE presence record** (sent by the scanner per staff member per IST hour):
`employee_id, beacon_mac, hour_bucket, ping_count, avg_rssi, min_rssi, max_rssi, first_ping_at, last_ping_at, minutes_present`

---

## 9. Output data

### 9.1 Visit record (one per closed track)

| Field | Type | Description |
|---|---|---|
| `camera_id` | text | Device that produced the record |
| `store_id`, `org_id` | uuid | Location identifiers from config |
| `is_employee` | bool | Matched to a staff beacon |
| `beacon_mac` | text / null | Beacon MAC if staff |
| `zone_name` | text / null | Majority zone (§7) |
| `section_name` | null | Reserved. Always null in this version |
| `gender` | enum | `male` / `female` / `unknown` |
| `age_bracket` | enum | `child` / `young_adult` / `adult` / `senior` / `unknown` |
| `entered_at` | ISO-8601 UTC | Time the track was created |
| `exited_at` | ISO-8601 UTC | Time the track was closed (includes ~1 s of the missing grace period) |
| `dwell_seconds` | int | `exited_at − entered_at`, floored, ≥ 2 |

Example:

```json
{
  "camera_id": "store01-cam1",
  "store_id": "<uuid>", "org_id": "<uuid>",
  "beacon_mac": null, "is_employee": false,
  "zone_name": "entrance", "section_name": null,
  "age_bracket": "young_adult", "gender": "female",
  "entered_at": "2026-10-15T08:12:03.512+00:00",
  "exited_at":  "2026-10-15T08:14:41.020+00:00",
  "dwell_seconds": 157
}
```

### 9.2 Local buffer

`$BUFFER_DB`: the SQLite file set by `DB_PATH` in `detector.py`, under `~/retail/` (WAL mode). Set it once in your shell, e.g. `BUFFER_DB=~/retail/<buffer-file>.db`.

Schema:

```sql
CREATE TABLE pending_tracks (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  payload    TEXT NOT NULL,   -- the JSON record above
  created_at TEXT NOT NULL    -- UTC time the record was buffered
);
```

Rows are removed only after a confirmed upload, so this table holds **all records not yet uploaded**. Upload batch size is 500. On clean shutdown, open tracks are closed, written and uploaded.

### 9.3 Derived fields commonly used downstream

| Field | Definition |
|---|---|
| Hour bucket | IST hour of `entered_at` |
| Day part | IST hour of entry: morning < 12, afternoon < 17, evening < 21, otherwise night |
| Dwell band | < 2 min (< 120 s) · 2–5 min (120–299) · 5–15 min (300–899) · > 15 min (≥ 900) |

### 9.4 Live stats endpoint

`GET http://<device-ip>:8080/stats` returns:

```json
{"customers": 3, "staff_cam": 1, "staff_ble": 1, "active_tracks": 5,
 "buffer_size": 42, "ist": "14:05:11",
 "recent_tracks": [{"track_id": 17, "role": "CUST", "gender": "female",
   "age_bracket": "adult", "zone": "entrance", "dwell": 34, "entered_at_ist": "14:04:30"}]}
```

`GET /stream` is an MJPEG stream of the annotated frames.

---

## 10. Inference from test-run output

**Run:** `detector.py` v2.6 on a fresh device, no remote connection (offline test mode), 15 Oct 2026.

| Observed output | Inference |
|---|---|
| `np_outputs received, 3 tensor(s)` every frame | Pose-model firmware is loaded and inference is running on the camera chip |
| Empty scene: raw scores **0.037 – 0.070**, all `< DETECTION_CONF 0.2 — skipped` | Background clutter produces weak person-like responses, but they're well under the threshold. **An empty scene creates no phantom records.** The 0.20 threshold leaves a ~3× margin over observed noise |
| `Model mivolo_d1_224 created, param count: 27432414` → `MiVolo ready` | Age and gender estimation is active. Without it, every record would be `unknown` |
| `Zone fetch failed (using disk cache)` | No zones available, so `zone_name` would be `null` for every record |
| `Employee cache: 0 employee(s)` | No staff roster, so **staff would be recorded as non-staff** |
| Remote-call failures (heartbeat, command poll) | Expected offline. Records accumulate in `pending_tracks` |

**What this run doesn't establish:** detection recall on real people, count accuracy, or age/gender accuracy in this camera placement. **No visit records were produced**, so there are no footfall or demographic figures to draw conclusions from yet.

**Recommended next step:** run a ground-truth session. Manually count people (and note gender and age band) for 1–2 hours, then compare with the records (§12, recipe E). Use the resulting ratio as a calibration factor until accuracy is validated.

---

## 11. Data-quality caveats & biases

### 11.1 Records are appearances, not unique people

- Leaving the view for more than ~4 s, long occlusion, or losing keypoints (turning away, crowding) creates **a new record** for the same person, so totals are **overcounted**.
- With multiple cameras, there's **no cross-camera de-duplication**.
- People visible for less than 2 s, or matched in fewer than 10 frames, are **dropped**, so quick passers-by are **undercounted**.
- **Use counts as a traffic index for comparisons over time, not as an absolute headcount**, until calibrated.

### 11.2 Duplicate records from track revival (known bug)

When a track is closed (missing > 15 frames), it's written as a record **and** placed in the re-ID buffer. If the person reappears within ~3 s, the same track is revived. When it closes again, it's **written a second time** with the **same `entered_at`** and a later `exited_at`.

→ **De-duplicate on `(camera_id, entered_at)` and keep the max `exited_at`** (recipe A).

### 11.3 Demographic biases

| Bias | Cause | Handling |
|---|---|---|
| Short visits → `unknown` | A label needs ≥ 2 agreeing votes, about 8 s or more in view | Compare mixes within the same dwell band. Report the known share |
| Higher `unknown` at busy times | Inference queue (size 8) drops requests under load | Check `unknown` rate vs hourly volume |
| `young_adult` inflation | If the model returns no age, the code substitutes **25 years** | Treat a sudden `young_adult` spike with suspicion |
| Camera angle and lighting | Overhead views and poor light degrade face crops | Don't compare demographic mixes across cameras without calibration |
| Binary gender | The model predicts male/female only | Report as an "estimated gender split", in aggregate only |

Compute shares **over known labels**, and always report the known share alongside them.

### 11.4 Dwell time

- It's time **in one camera's field of view**, not total visit time.
- It's split across records when a track is lost (§11.1). Combine with recipe A before averaging.
- It includes ~1 s of grace at exit.
- Hour attribution uses `entered_at` only. A 40-minute visit starting at 13:55 falls entirely in the 13:00 hour.

### 11.5 Staff flag

- Staff without a working or charged beacon are recorded as **non-staff**.
- The beacon is matched to the longest-lived nearby track, so a customer standing next to staff can occasionally be flagged instead.
- For staff time, prefer the **BLE presence records** (§8). The camera flag only covers time the person was both visible and matched.

### 11.6 Data gaps

- No records for a period can mean **zero traffic, device downtime or a camera fault**. Check device uptime or buffer state before treating a gap as zero.
- `created_at − exited_at` (buffer time vs visit time) shows ingestion lag. Records can arrive many hours after the visit, because uploads run 3×/day.
- Changing any parameter in §13 changes metric definitions. **Log the date of every tuning change** and mark breaks in trend lines.

---

## 12. Analysis recipes

These run directly on the device buffer (`sqlite3 $BUFFER_DB`), or on any table with the same fields after flattening the JSON.

**A. De-duplicated non-staff visits**
```sql
WITH t AS (
  SELECT json_extract(payload,'$.camera_id')   AS cam,
         json_extract(payload,'$.entered_at')  AS entered_at,
         json_extract(payload,'$.exited_at')   AS exited_at,
         json_extract(payload,'$.dwell_seconds') AS dwell,
         json_extract(payload,'$.gender')      AS gender,
         json_extract(payload,'$.age_bracket') AS age,
         json_extract(payload,'$.zone_name')   AS zone
  FROM pending_tracks
  WHERE json_extract(payload,'$.is_employee') = 0
)
SELECT cam, entered_at, MAX(exited_at) AS exited_at, MAX(dwell) AS dwell,
       gender, age, zone
FROM t GROUP BY cam, entered_at;
```

**B. Hourly visits (IST)**
```sql
SELECT strftime('%Y-%m-%d %H:00', json_extract(payload,'$.entered_at'), '+5 hours', '+30 minutes') AS hour_ist,
       COUNT(DISTINCT json_extract(payload,'$.camera_id') || json_extract(payload,'$.entered_at')) AS visits
FROM pending_tracks
WHERE json_extract(payload,'$.is_employee') = 0
GROUP BY hour_ist ORDER BY hour_ist;
```

**C. Unknown-label rate by dwell band** (checks the §11.3 bias)
```sql
SELECT CASE WHEN d < 120 THEN '<2m' WHEN d < 300 THEN '2-5m'
            WHEN d < 900 THEN '5-15m' ELSE '>15m' END AS band,
       COUNT(*) AS n,
       ROUND(AVG(g = 'unknown'), 3) AS gender_unknown_rate,
       ROUND(AVG(a = 'unknown'), 3) AS age_unknown_rate
FROM (SELECT json_extract(payload,'$.dwell_seconds') d,
             json_extract(payload,'$.gender') g,
             json_extract(payload,'$.age_bracket') a
      FROM pending_tracks WHERE json_extract(payload,'$.is_employee') = 0)
GROUP BY band;
```

**D. Gender share over known labels**
```sql
SELECT ROUND(AVG(g = 'female'), 3) AS female_share_of_known,
       COUNT(*) AS known_n
FROM (SELECT json_extract(payload,'$.gender') g FROM pending_tracks
      WHERE json_extract(payload,'$.is_employee') = 0)
WHERE g <> 'unknown';
```

**E. Ground-truth comparison**
```sql
SELECT COUNT(*) FROM pending_tracks
WHERE json_extract(payload,'$.is_employee') = 0
  AND json_extract(payload,'$.entered_at') BETWEEN '<start_utc_iso>' AND '<end_utc_iso>';
```
Calibration ratio = system count ÷ manual count (run recipe A first to remove duplicates).

**F. Ingestion lag**
```sql
SELECT AVG((julianday(created_at) - julianday(json_extract(payload,'$.exited_at'))) * 86400) AS avg_lag_s
FROM pending_tracks;
```

---

## 13. Parameters reference

| Constant | Default | Effect of raising | Effect of lowering |
|---|---|---|---|
| `FRAME_W × FRAME_H` | 640 × 640 | — | — |
| `TARGET_FPS` | 15 | — | — |
| `DETECTION_CONF` | 0.20 | Fewer false positives, misses distant or partial people | More recall, risk of phantoms (don't go below ~0.15) |
| `KP_CONF` | 0.10 | Tighter boxes, more dropped people | Looser boxes |
| `MIN_BOX_PX` | 15 | Ignores small/far people | Counts far people |
| `IOU_GATE` / `DIST_GATE` | 0.05 / 200 px | Looser matching (more merges) | Stricter matching (more splits) |
| `MAX_MISSING` | 15 frames | Fewer split tracks, later exits | More splits |
| `REID_FRAMES` | 45 frames | More revivals (more §11.2 duplicates) | More new records |
| `MIN_TRACK_AGE` | 10 frames | Fewer short records | More short records |
| `MIN_TRACK_SECS` | 2 s | Ignores passers-by | Counts passers-by |
| `MIVOLO_REINFER_FRAMES` | 60 | Less CPU, more `unknown` | Faster labels, more CPU |
| `MIVOLO_MIN_CONF` | 0.45 | More `unknown` gender | More (less certain) labels |
| `MIN_VOTES` | 2 | More stable, more `unknown` | Faster, noisier labels |
| `BLE_RSSI_STRONG_MAX` / `MEDIUM_MAX` | 55 / 75 | Wider staff matching radius | Narrower radius |
| `MIN_PERSIST_FRAMES` | 120 | Fewer wrong staff flags | Faster staff flagging |
| `BLE_STALE_SECS` | 15 | — | — |
| `upload_times_utc` (config) | 03:30, 08:30, 14:30 | — | — |

---

## 14. Setup & running

### 14.1 Install (Raspberry Pi OS Bookworm 64-bit, Pi 5)

```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y imx500-all python3-picamera2 python3-opencv python3-venv git
sudo reboot

python3 -m venv --system-site-packages ~/edge-env
source ~/edge-env/bin/activate
pip install "numpy<2"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics requests schedule flask bleak
pip install git+https://github.com/WildChlamydia/MiVOLO.git
```

- **Pose model** is installed by `imx500-all` at `/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk`.
- **MiVolo checkpoint:** download `model_imdb_cross_person_4.22_99.46.pth.tar` (Hugging Face repo `Genius-Society/MiVOLO`) and save it as `~/models/mivolo_checkpoint.pth.tar`.
- Keep `numpy < 2` (picamera2 compatibility). Use the apt `python3-opencv`, **not** pip `opencv-python`, whose bundled Qt crashes the preview window.

### 14.2 Configuration (`~/retail/config.json`, `chmod 600`)

```json
{
  "camera_id": "store01-cam1",
  "device_token": "<device secret>",
  "org_id": "<uuid>",
  "store_id": "<uuid>",
  "edge_function_url": "<API base URL>",
  "ble_pings_edge_url": "<BLE presence endpoint URL>",
  "mivolo_model": "/home/<user>/models/mivolo_checkpoint.pth.tar"
}
```

Optional keys: `upload_times_utc` (list of `"HH:MM"`) and `preview_port` (default 8080).
**Offline test mode:** set `edge_function_url` to `http://127.0.0.1:9`. Detection runs normally and all records stay in the buffer database.

### 14.3 Files on the device (`~/retail/`)

| File | Purpose |
|---|---|
| `detector.py` | Main pipeline |
| `person_detect.py` | Minimal live camera demo that draws person detections |
| `blescanner.py` | Staff beacon scanner |
| `launch_detector.sh` | Activates the venv and runs the detector (`--test` checks imports) |
| `config.json` | Device configuration |
| the buffer database | Buffered visit records |
| `zone_cache.json` | Cached zone definitions |
| `employees_cache.json` / `employees_updated.flag` | Staff roster plus hot-reload trigger |
| `ble_state.json` | Staff currently present (scanner → detector) |

### 14.4 Run & monitor

| Task | Command |
|---|---|
| Simple person-detection demo | `python ~/retail/person_detect.py` (Q or Esc to quit) |
| Run headless | `python ~/retail/detector.py` |
| Run with local window | `python ~/retail/detector.py --preview` (Q quit · U upload now · R reload zones) |
| Live view | `http://<device-ip>:8080` |
| Pending records | `sqlite3 $BUFFER_DB "select count(*) from pending_tracks;"` |
| Latest records | `sqlite3 $BUFFER_DB "select payload from pending_tracks order by id desc limit 10;"` |
| Service logs | `journalctl -u <detector-service> -n 100 --no-pager` |
| Camera self-test (no Python) | `rpicam-hello -t 0s --post-process-file /usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json` |

**Log levels:** `DEBUG` shows per-frame raw scores (useful for threshold tuning). `INFO` shows detections, model status, a once-a-minute summary (`Active | Visible | Staff | Buffer`) and errors.

---

## 15. Glossary

| Term | Meaning |
|---|---|
| **Track / record** | One continuous appearance of one person in one camera's view |
| **Dwell** | Duration of a track in seconds |
| **Keypoint** | One of 17 body landmarks (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) |
| **NMS** | Non-maximum suppression: removes overlapping duplicate detections |
| **IoU** | Intersection-over-union of two boxes (0–1) |
| **Re-ID** | Reconnecting a briefly lost person to their previous track |
| **Vote** | One age/gender prediction for a track. Labels are committed by majority |
| **Zone** | User-defined area of the camera frame, in pixel coordinates |
| **BLE / iBeacon** | Bluetooth Low Energy tag format worn by staff |
| **RSSI** | Received signal strength (dBm). Closer to 0 means nearer |
| **IST** | India Standard Time, UTC+05:30 |
=======
# Age-Gender_Edge
The repo has codes for age gender prediction on edge device
>>>>>>> e308f2b6db564a17a93eefc397ed75f6a3c9d3cd
