# Device File Paths: Reference

These are the file paths defined at the top of `detector.py`:

```python
CONFIG_PATH           = Path.home() / "retail" / "config.json"
DB_PATH               = Path.home() / "retail" / "vidar.db"
BLE_STATE_PATH        = Path.home() / "retail" / "ble_state.json"
EMPLOYEES_CACHE_PATH  = Path.home() / "retail" / "employees_cache.json"
EMPLOYEES_RELOAD_FLAG = Path.home() / "retail" / "employees_updated.flag"
ZONE_CACHE_PATH       = Path.home() / "retail" / "zone_cache.json"

IMX500_MODEL = "/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk"
```

`Path.home()` is the Pi user's home directory, e.g. `/home/<pi-user>`. So `~/retail/` means `/home/<pi-user>/retail/`.

In the README, `$BUFFER_DB` refers to `~/retail/vidar.db`.

---

## Summary

| Path | What it is | Written by | Format | Relevant for analysis? |
|---|---|---|---|---|
| `~/retail/config.json` | Device settings: camera ID, store/org IDs, API URL, device token, model path | Set up manually | JSON | Only for identifiers (`camera_id`, `store_id`, `org_id`). Contains a secret token, so do not share the file |
| **`~/retail/vidar.db`** | **Visit records:** one row per person visit, held until uploaded | `detector.py` | SQLite | **Yes, this is the main data file** (see below) |
| `~/retail/ble_state.json` | Live snapshot of staff beacons currently nearby | `blescanner.py`, every 2 s | JSON | No: overwritten constantly, no history |
| `~/retail/employees_cache.json` | Local copy of the staff roster | `blescanner.py`, hourly | JSON list | Lookup only: maps `beacon_mac` → staff member. Contains personal data |
| `~/retail/employees_updated.flag` | Empty signal file telling the detector to reload the roster | `blescanner.py` | Empty file (modified time only) | No |
| `~/retail/zone_cache.json` | Saved zone definitions | `detector.py` | JSON list | Yes, as reference: defines what each `zone_name` covers |
| `/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk` | Person-detection (pose) model that runs on the camera chip | `sudo apt install imx500-all` | Binary model package | No: model file, not data |

---

## File details

### `~/retail/vidar.db`: visit records (main data file)

SQLite database in WAL mode, with one table:

```sql
CREATE TABLE pending_tracks (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  payload    TEXT NOT NULL,   -- JSON visit record
  created_at TEXT NOT NULL    -- UTC time the record was buffered
);
```

Example `payload`:

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

**Important:** this is a **temporary buffer, not an archive**. Rows are deleted after each successful upload (scheduled at 03:30, 08:30 and 14:30 UTC, plus on clean shutdown). At any moment, the file holds only records that haven't been uploaded yet. Historical data must come from the uploaded data store.

The filename `vidar.db` is fixed by `DB_PATH` in the code.

### `~/retail/config.json`: device settings

```json
{
  "camera_id": "store01-cam1",
  "device_token": "<secret>",
  "org_id": "<uuid>",
  "store_id": "<uuid>",
  "edge_function_url": "<API base URL>",
  "ble_pings_edge_url": "<BLE presence endpoint URL>",
  "mivolo_model": "/home/<pi-user>/models/mivolo_checkpoint.pth.tar"
}
```

Only the identifier fields are useful for analysis. **Never share the actual file.** It contains the device's secret token.

### `~/retail/ble_state.json`: staff currently present

```json
{"count": 1,
 "employees": [{"employee_id": "...", "name": "...", "beacon_mac": "...", "rssi": -48}],
 "ts": "2026-10-15T08:12:03+00:00"}
```

Rewritten every 2 seconds. The detector ignores it if `ts` is older than 15 seconds. It has no history, so it isn't useful for analysis.

### `~/retail/employees_cache.json`: staff roster

```json
[{"employee_id": "...", "name": "...", "gender": "female",
  "age_bracket": "adult", "beacon_mac": "AA:BB:CC:DD:EE:FF"}]
```

Refreshed hourly. Use it to join `beacon_mac` in visit records to a staff member. It holds employee personal data, so handle it accordingly.

### `~/retail/employees_updated.flag`

An empty file. Updating its modified time tells the detector to reload `employees_cache.json` without restarting. It holds no data.

### `~/retail/zone_cache.json`: zone definitions

```json
[{"name": "entrance", "points": [[192,332],[68,434],[257,439],[250,333]]},
 {"name": "checkout", "x1": 320, "y1": 0, "x2": 640, "y2": 300}]
```

Coordinates are in **camera pixels (640×640 frame)**, as polygons (`points`) or rectangles (`x1, y1, x2, y2`). Use it to see what area each `zone_name` covers. **If this file is empty or missing, every visit record has `zone_name = null`.**

### `/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk`: detection model

The HigherHRNet multi-person pose model (17 body keypoints per person). It is loaded onto the IMX500 camera's AI chip at startup and installed by `sudo apt install imx500-all`. This is a model file, not data.

---

## Accessing the data

**View the latest records on the device:**

```bash
sqlite3 ~/retail/vidar.db "select payload from pending_tracks order by id desc limit 10;"
```

**Count pending (un-uploaded) records:**

```bash
sqlite3 ~/retail/vidar.db "select count(*) from pending_tracks;"
```

**Export a copy for analysis:**

```bash
cp ~/retail/vidar.db ~/data_export_$(date +%F).db
```

Copy the file rather than querying the live one while the detector is writing to it. For a guaranteed-complete copy, stop the detector first: pending writes are stored in the `-wal` file next to the database until they're merged in.

**Flatten records into columns (SQLite):**

```sql
SELECT json_extract(payload,'$.camera_id')     AS camera_id,
       json_extract(payload,'$.entered_at')    AS entered_at,
       json_extract(payload,'$.exited_at')     AS exited_at,
       json_extract(payload,'$.dwell_seconds') AS dwell_seconds,
       json_extract(payload,'$.gender')        AS gender,
       json_extract(payload,'$.age_bracket')   AS age_bracket,
       json_extract(payload,'$.zone_name')     AS zone_name,
       json_extract(payload,'$.is_employee')   AS is_employee
FROM pending_tracks;
```
