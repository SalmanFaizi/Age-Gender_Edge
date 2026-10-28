#!/usr/bin/env python3
"""IMX500 person detection + ByteTrack (supervision) + bbox / track-ID overlay.

Detection runs on the IMX500 sensor. The camera main stream can be configured at a
higher resolution than the model's fixed input tensor; IMX500 coordinate conversion
maps detections into that stream's pixel coordinates. Tracking and drawing run on
the Pi inside the camera pre_callback, so detections and pixels stay aligned.
"""
import argparse
import io
import json
import math
import queue
import sys
import threading
import time
import traceback
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import requests
import supervision as sv

# supervision removed ByteTrack in 0.31.0 (deprecated since 0.28, now lives in Roboflow's
# separate `trackers` package, which needs numpy>=2). Install with: pip install "supervision<0.31"
if not hasattr(sv, "ByteTrack"):
    sys.exit('This supervision version has no ByteTrack. Run: pip install "supervision<0.31"')

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics, postprocess_nanodet_detection

FONT = cv2.FONT_HERSHEY_SIMPLEX
PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49),
    (72, 249, 10), (146, 204, 23), (61, 219, 134), (26, 147, 52), (0, 212, 187),
    (44, 153, 168), (0, 194, 255), (52, 69, 147), (100, 115, 255), (0, 24, 236),
    (132, 56, 255), (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]

# Standard 80-class COCO labels (YOLO order). Used when the model file has no labels of its own.
COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]

# Latest tracked result, replaced as a whole list each update (safe to read from the draw step).
# Each item: (x1, y1, x2, y2, track_id, conf, label)
overlay_tracks = []
stats = {"fps": 0.0, "t_last": None}
track_windows = {}
completed_track_ids = set()
mivolo_queue = queue.Queue()


def load_config(path):
    with open(path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError("config root must be a JSON object")

    model_type = config.get("model_type")
    models = config.get("models")
    if model_type not in ("yolo", "default") or not isinstance(models, dict):
        raise ValueError("config must define model_type as 'yolo' or 'default' and a models object")
    if not isinstance(models.get(model_type), str) or not models[model_type]:
        raise ValueError(f"config is missing models.{model_type}")

    classes = config.get("classes", ["person"])
    if not isinstance(classes, list) or not classes or any(
        not isinstance(name, str) or not name for name in classes
    ):
        raise ValueError("classes must be a non-empty list of label names")

    image_size = config.get("image_size")
    if (
        not isinstance(image_size, list)
        or len(image_size) != 2
        or any(type(value) is not int or value <= 0 for value in image_size)
    ):
        raise ValueError("image_size must be [width, height] with positive integers")

    zone = config.get("detection_zone")
    if not isinstance(zone, dict) or not isinstance(zone.get("points"), list) or len(zone["points"]) < 3:
        raise ValueError("detection_zone.points must contain at least three [x, y] points")
    if any(
        not isinstance(point, list)
        or len(point) != 2
        or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in point)
        for point in zone["points"]
    ):
        raise ValueError("each detection zone point must contain two finite coordinates")

    api_url = config.get("mivolo_api_url")
    if not isinstance(api_url, str) or not api_url.startswith(("http://", "https://")):
        raise ValueError("mivolo_api_url must be an HTTP(S) URL")
    timeout = config.get("mivolo_timeout_seconds", 120)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("mivolo_timeout_seconds must be a positive number")
    return config


def point_in_zone(box):
    x1, y1, x2, y2 = box
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    polygon = np.asarray(config["detection_zone"]["points"], dtype=np.float32)
    return cv2.pointPolygonTest(polygon, center, False) >= 0


def post_mivolo():
    while True:
        job = mivolo_queue.get()
        try:
            if job is None:
                return
            image, box, track_id = job
            print(
                f"[mivolo] sending track {track_id}: full-frame JPEG={len(image)} bytes, "
                f"body_bbox={box}",
                flush=True,
            )
            response = requests.post(
                config["mivolo_api_url"],
                files={"image": ("frame.jpg", io.BytesIO(image), "image/jpeg")},
                data={"body_bbox": json.dumps(box), "track_id": str(track_id)},
                timeout=config.get("mivolo_timeout_seconds", 120),
            )
            response.raise_for_status()
            print(f"[mivolo] track {track_id}: {response.json()}")
        except requests.RequestException as exc:
            print(f"[mivolo] request failed: {exc}", file=sys.stderr)
        except ValueError as exc:
            print(f"[mivolo] server returned invalid JSON: {exc}", file=sys.stderr)
        finally:
            mivolo_queue.task_done()


@lru_cache
def get_labels():
    labels = intrinsics.labels
    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels


_debug_times = {}


def debug_log(key, message, interval=1.0):
    now = time.monotonic()
    if now - _debug_times.get(key, 0.0) < interval:
        return
    _debug_times[key] = now
    print(f"[debug] {message}", flush=True)


def debug_dump(np_outputs, boxes, scores, classes, candidates, rejected):
    labels = get_labels()
    raw_top = []
    for i in np.argsort(-np.asarray(scores, dtype=float))[:5]:
        c = int(classes[i])
        raw_top.append(
            (
                c,
                labels[c] if 0 <= c < len(labels) else "?",
                round(float(scores[i]), 3),
                [round(float(value), 1) for value in boxes[i]],
            )
        )
    debug_log(
        "inference",
        f"sensor output shapes={[np.shape(output) for output in np_outputs]}; "
        f"bbox_order={intrinsics.bbox_order}, normalized={intrinsics.bbox_normalization}, "
        f"postprocess={intrinsics.postprocess!r}, threshold={args.threshold:.2f}; "
        f"top model outputs after bbox order/normalization "
        f"(class_id, label, score, model_box)={raw_top}; "
        f"passed score/class filters={len(candidates)} "
        f"(below_threshold={rejected['threshold']}, unwanted_class={rejected['class']}); "
        f"camera-frame detections [x1,y1,x2,y2] in {config['image_size']} frame="
        f"{candidates or 'none'}",
    )


def parse_detections(metadata: dict):
    """Turn the IMX500 output tensors into an sv.Detections in ISP-output pixel coordinates.

    Returns None when this frame carries no fresh inference output, so the tracker is not
    fed stale (duplicate) detections.
    """
    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    if np_outputs is None:
        if args.debug:
            debug_log("no-output", "no fresh IMX500 inference output on this camera frame")
        return None

    input_w, input_h = imx500.get_input_size()
    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = postprocess_nanodet_detection(
            outputs=np_outputs[0], conf=args.threshold, iou_thres=args.iou, max_out_dets=args.max_detections
        )[0]
        from picamera2.devices.imx500.postprocess import scale_boxes

        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if intrinsics.bbox_normalization:
            boxes = boxes / input_h
        if intrinsics.bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]

    xyxy, confs, class_ids = [], [], []
    candidates = []
    rejected = {"threshold": 0, "class": 0}
    for box, score, category in zip(boxes, scores, classes):
        if score < args.threshold:
            rejected["threshold"] += 1
            continue
        if int(category) not in keep_ids:
            rejected["class"] += 1
            continue
        x, y, w, h = imx500.convert_inference_coords(box, metadata, picam2)
        frame_box = [x, y, x + w, y + h]
        class_id = int(category)
        xyxy.append(frame_box)
        confs.append(float(score))
        class_ids.append(class_id)
        candidates.append(
            (
                class_id,
                get_labels()[class_id] if 0 <= class_id < len(get_labels()) else "?",
                round(float(score), 3),
                [round(float(value), 1) for value in frame_box],
            )
        )

    if args.debug:
        debug_dump(np_outputs, boxes, scores, classes, candidates, rejected)

    if not xyxy:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        confidence=np.array(confs, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
    )


def update_tracks(metadata: dict):
    global overlay_tracks
    detections = parse_detections(metadata)
    if detections is None:  # nothing new from the sensor: leave tracker and overlay as they are
        return False, []

    # Always update, even with zero detections, so lost tracks age out correctly.
    tracked = tracker.update_with_detections(detections)

    labels = get_labels()
    items = []
    if len(tracked) and tracked.tracker_id is not None:
        for (x1, y1, x2, y2), conf, cls, tid in zip(
            tracked.xyxy, tracked.confidence, tracked.class_id, tracked.tracker_id
        ):
            if tid < 0:
                continue
            items.append((int(x1), int(y1), int(x2), int(y2), int(tid), float(conf), labels[int(cls)]))
    valid_items = [item for item in items if point_in_zone(item[:4])]
    overlay_tracks = valid_items
    if args.debug:
        excluded = [item for item in items if item not in valid_items]
        visible = [
            {
                "id": item[4],
                "label": item[6],
                "confidence": round(item[5], 3),
                "box": item[:4],
            }
            for item in valid_items
        ]
        debug_log(
            "tracking",
            f"ByteTrack outputs={len(items)}, inside_detection_zone={len(valid_items)}, "
            f"outside_zone_ids={[item[4] for item in excluded]}; visible tracks={visible or 'none'}",
        )

    now = time.monotonic()
    if stats["t_last"] is not None and now > stats["t_last"]:
        inst = 1.0 / (now - stats["t_last"])
        stats["fps"] = inst if stats["fps"] == 0.0 else 0.9 * stats["fps"] + 0.1 * inst
    stats["t_last"] = now
    return True, valid_items


def collect_track_frames(request, items):
    for window in track_windows.values():
        window["target_frames"] += 1
        window["missed_frames"] += 1

    frame_bytes = None
    if items:
        with MappedArray(request, "main") as mapped:
            # XBGR8888 is exposed as RGBA bytes on the Pi's little-endian CPU.
            frame_bgr = cv2.cvtColor(mapped.array, cv2.COLOR_RGBA2BGR)
            ok, encoded = cv2.imencode(
                ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90]
            )
        if not ok:
            raise RuntimeError("OpenCV failed to encode the full camera frame")
        frame_bytes = encoded.tobytes()

    for x1, y1, x2, y2, track_id, confidence, _label in items:
        if track_id in completed_track_ids:
            continue
        window = track_windows.get(track_id)
        if window is None:
            window = {
                "target_frames": 1,
                "missed_frames": 0,
                "valid_frames": 0,
                "best_sample": None,
            }
            track_windows[track_id] = window
        else:
            window["missed_frames"] = 0

        width, height = config["image_size"]
        box = [
            max(0, min(width, x1)),
            max(0, min(height, y1)),
            max(0, min(width, x2)),
            max(0, min(height, y2)),
        ]
        area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
        sample = (area, confidence, frame_bytes, box)
        window["valid_frames"] += 1
        if window["best_sample"] is None or sample[:2] > window["best_sample"][:2]:
            window["best_sample"] = sample

    for track_id, window in list(track_windows.items()):
        if window["target_frames"] >= args.target_frames:
            reason = "target reached"
        elif window["missed_frames"] > args.track_buffer:
            reason = "track lost"
        else:
            continue

        required = math.ceil(args.target_frames * args.minimum_valid_ratio)
        if window["valid_frames"] >= required:
            _area, _confidence, image, box = window["best_sample"]
            mivolo_queue.put((image, box, track_id))
            print(
                f"[mivolo] queued track {track_id} ({reason}): "
                f"{window['valid_frames']}/{window['target_frames']} valid frames"
            )
        else:
            print(
                f"[mivolo] skipped track {track_id} ({reason}): "
                f"{window['valid_frames']}/{window['target_frames']} valid frames "
                f"(minimum {required})"
            )
        completed_track_ids.add(track_id)
        del track_windows[track_id]


def draw(request, stream="main"):
    """Draw boxes, track IDs and a small HUD directly onto the ISP output."""
    items = overlay_tracks
    with MappedArray(request, stream) as m:
        zone = np.asarray(config["detection_zone"]["points"], dtype=np.int32)
        cv2.polylines(m.array, [zone], True, (255, 0, 0), 2)
        for x1, y1, x2, y2, tid, conf, label in items:
            color = PALETTE[tid % len(PALETTE)]
            cv2.rectangle(m.array, (x1, y1), (x2, y2), color, 2)

            text = f"#{tid} {label} {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(text, FONT, 0.5, 1)
            box_h = th + baseline + 4
            top = y1 - box_h if y1 - box_h >= 0 else y1  # put label inside box if no room above
            cv2.rectangle(m.array, (x1, top), (x1 + tw + 4, top + box_h), color, cv2.FILLED)
            cv2.putText(m.array, text, (x1 + 2, top + th + 1), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(m.array, f"tracks: {len(items)}   {stats['fps']:.1f} fps", (10, 22), FONT, 0.6, (0, 255, 255), 2)

        if intrinsics.preserve_aspect_ratio:
            b_x, b_y, b_w, b_h = imx500.get_roi_scaled(request)
            cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15), FONT, 0.5, (255, 0, 0), 1)
            cv2.rectangle(m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0))


def on_frame(request):
    """Runs in the camera thread for every frame: detect -> track -> draw."""
    try:
        fresh, items = update_tracks(request.get_metadata())
        if fresh:
            collect_track_frames(request, items)
        if not args.headless:
            draw(request)
    except Exception:
        traceback.print_exc()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("detection_config.json"),
        help="Path to the detection and MiVOLO configuration JSON",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path of the model",
        default=None,
    )
    parser.add_argument("--fps", type=int, help="Frames per second (default: the network's inference rate)")
    parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction, help="Normalize bbox")
    parser.add_argument(
        "--bbox-order", choices=["yx", "xy"], default="yx", help="Set bbox order yx -> (y0, x0, y1, x1) xy -> (x0, y0, x1, y1)"
    )
    # Keep the detector threshold LOW: ByteTrack uses the low-confidence boxes for its second
    # association pass. New tracks only start from boxes above --track-thresh.
    parser.add_argument("--threshold", type=float, default=0.1, help="Detection threshold (low, for ByteTrack)")
    parser.add_argument("--iou", type=float, default=0.65, help="Set iou threshold")
    parser.add_argument("--max-detections", type=int, default=10, help="Set max detections")
    parser.add_argument("--ignore-dash-labels", action=argparse.BooleanOptionalAction, help="Remove '-' labels ")
    parser.add_argument("--postprocess", choices=["", "nanodet"], default=None, help="Run post process of type")
    parser.add_argument(
        "-r",
        "--preserve-aspect-ratio",
        action=argparse.BooleanOptionalAction,
        help="preserve the pixel aspect ratio of the input tensor",
    )
    parser.add_argument("--labels", type=str, help="Path to the labels file")
    parser.add_argument("--print-intrinsics", action="store_true", help="Print JSON network_intrinsics then exit")

    # Tracking / app options
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Label names to track (overrides classes in the detection config)",
    )
    parser.add_argument("--track-thresh", type=float, default=0.4, help="ByteTrack: confidence needed to start/confirm a track")
    parser.add_argument("--track-buffer", type=int, default=30, help="ByteTrack: frames to keep a lost track alive")
    parser.add_argument("--match-thresh", type=float, default=0.8, help="ByteTrack: matching threshold")
    parser.add_argument("--headless", action="store_true", help="No preview window; print tracks to the console")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print model, camera-frame detection, tracking and zone diagnostics",
    )
    parser.add_argument("--target-frames", type=int, default=20, help="Inference frames collected per track")
    parser.add_argument(
        "--minimum-valid-ratio",
        type=float,
        default=0.3,
        help="Minimum fraction of target frames with a zone-valid detection before MiVOLO analysis",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    try:
        config = load_config(args.config)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Unable to load detection config {args.config}: {exc}", file=sys.stderr)
        sys.exit(1)
    if args.target_frames <= 0 or not 0 < args.minimum_valid_ratio <= 1:
        print("--target-frames must be positive and --minimum-valid-ratio must be in (0, 1]", file=sys.stderr)
        sys.exit(1)

    # This must be called before instantiation of Picamera2
    imx500 = IMX500(args.model or config["models"][config["model_type"]])
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    elif intrinsics.task != "object detection":
        print("Network is not an object detection task", file=sys.stderr)
        sys.exit(1)

    # Override intrinsics from args
    for key, value in vars(args).items():
        if key == "labels" and value is not None:
            with open(value, "r") as f:
                intrinsics.labels = f.read().splitlines()
        elif hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    # Defaults
    if intrinsics.labels is None:  # e.g. the YOLO .rpk files ship without labels
        intrinsics.labels = COCO80
    intrinsics.update_with_defaults()

    if args.print_intrinsics:
        print(intrinsics)
        sys.exit(0)

    args.classes = args.classes or config["classes"]

    keep_ids = {i for i, name in enumerate(get_labels()) if name in args.classes}
    if not keep_ids:
        print(f"None of {args.classes} found in the model labels", file=sys.stderr)
        sys.exit(1)

    if args.debug:
        print(
            f"[debug] startup: model={args.model or config['models'][config['model_type']]!r} "
            f"model_type={config['model_type']!r}, classes={args.classes}, "
            f"class_ids={sorted(keep_ids)}, "
            f"image_size={config['image_size']}, zone={config['detection_zone']['points']}, "
            f"bbox_order={intrinsics.bbox_order}, normalized={intrinsics.bbox_normalization}, "
            f"threshold={args.threshold}, track_start_threshold={args.track_thresh}, "
            f"MiVOLO target={args.target_frames} inference frames, "
            f"minimum_valid={math.ceil(args.target_frames * args.minimum_valid_ratio)}/"
            f"{args.target_frames}",
            flush=True,
        )

    fps = args.fps or intrinsics.inference_rate or 30
    tracker = sv.ByteTrack(
        track_activation_threshold=args.track_thresh,
        lost_track_buffer=args.track_buffer,
        minimum_matching_threshold=args.match_thresh,
        frame_rate=int(fps),
    )

    mivolo_thread = threading.Thread(target=post_mivolo, daemon=True, name="mivolo-uploader")
    mivolo_thread.start()
    picam2 = Picamera2(imx500.camera_num)
    camera_config = picam2.create_preview_configuration(
        main={"size": tuple(config["image_size"]), "format": "XBGR8888"},
        controls={"FrameRate": fps},
        buffer_count=12,
    )

    imx500.show_network_fw_progress_bar()
    picam2.start(camera_config, show_preview=not args.headless)

    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()

    picam2.pre_callback = on_frame

    try:
        while True:
            time.sleep(1)
            if args.headless:
                print(f"{stats['fps']:.1f} fps | active track ids: {[t[4] for t in overlay_tracks]}")
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
        mivolo_queue.join()
        mivolo_queue.put(None)
        mivolo_thread.join()