#!/usr/bin/env python3
"""IMX500 person detection + ByteTrack (supervision) + bbox / track-ID overlay.

Detection runs on the IMX500 sensor. Tracking and drawing run on the Pi inside the
camera pre_callback, so the detections, the tracker update and the frame being drawn
all come from the same request (no lag between boxes and pixels).
"""
import argparse
import sys
import time
import traceback
from functools import lru_cache

import cv2
import numpy as np
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

# Latest tracked result, replaced as a whole list each update (safe to read from the draw step).
# Each item: (x1, y1, x2, y2, track_id, conf, label)
overlay_tracks = []
stats = {"fps": 0.0, "t_last": None}


@lru_cache
def get_labels():
    labels = intrinsics.labels
    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels


def parse_detections(metadata: dict):
    """Turn the IMX500 output tensors into an sv.Detections in ISP-output pixel coordinates.

    Returns None when this frame carries no fresh inference output, so the tracker is not
    fed stale (duplicate) detections.
    """
    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    if np_outputs is None:
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
    for box, score, category in zip(boxes, scores, classes):
        if score < args.threshold or int(category) not in keep_ids:
            continue
        x, y, w, h = imx500.convert_inference_coords(box, metadata, picam2)
        xyxy.append([x, y, x + w, y + h])
        confs.append(score)
        class_ids.append(int(category))

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
        return

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
    overlay_tracks = items

    now = time.monotonic()
    if stats["t_last"] is not None and now > stats["t_last"]:
        inst = 1.0 / (now - stats["t_last"])
        stats["fps"] = inst if stats["fps"] == 0.0 else 0.9 * stats["fps"] + 0.1 * inst
    stats["t_last"] = now


def draw(request, stream="main"):
    """Draw boxes, track IDs and a small HUD directly onto the ISP output."""
    items = overlay_tracks
    with MappedArray(request, stream) as m:
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
        update_tracks(request.get_metadata())
        if not args.headless:
            draw(request)
    except Exception:
        traceback.print_exc()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="Path of the model",
        default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk",
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
    parser.add_argument("--classes", nargs="+", default=["person"], help="Label names to track")
    parser.add_argument("--track-thresh", type=float, default=0.4, help="ByteTrack: confidence needed to start/confirm a track")
    parser.add_argument("--track-buffer", type=int, default=30, help="ByteTrack: frames to keep a lost track alive")
    parser.add_argument("--match-thresh", type=float, default=0.8, help="ByteTrack: matching threshold")
    parser.add_argument("--headless", action="store_true", help="No preview window; print tracks to the console")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # This must be called before instantiation of Picamera2
    imx500 = IMX500(args.model)
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
    if intrinsics.labels is None:
        with open("assets/coco_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()
    intrinsics.update_with_defaults()

    if args.print_intrinsics:
        print(intrinsics)
        sys.exit(0)

    keep_ids = {i for i, name in enumerate(get_labels()) if name in args.classes}
    if not keep_ids:
        print(f"None of {args.classes} found in the model labels", file=sys.stderr)
        sys.exit(1)

    fps = args.fps or intrinsics.inference_rate or 30
    tracker = sv.ByteTrack(
        track_activation_threshold=args.track_thresh,
        lost_track_buffer=args.track_buffer,
        minimum_matching_threshold=args.match_thresh,
        frame_rate=int(fps),
    )

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(controls={"FrameRate": fps}, buffer_count=12)

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=not args.headless)

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