#!/usr/bin/env python3
"""Show live person detections from the Raspberry Pi IMX500 camera."""

import logging
import signal
import threading
import time

import cv2
import numpy as np
from picamera2 import MappedArray, Picamera2
from picamera2.devices.imx500 import IMX500
from picamera2.devices.imx500.postprocess_highernet import postprocess_higherhrnet

MODEL_PATH = "/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk"
FRAME_WIDTH = 640
FRAME_HEIGHT = 640
KEYPOINT_CONFIDENCE = 0.10
PERSON_CONFIDENCE = 0.20
MIN_BOX_SIZE = 15

log = logging.getLogger("person_detect")
state = {"frame": None, "people": [], "lock": threading.Lock()}
stopping = threading.Event()


def _body_box(keypoints: np.ndarray) -> list[float] | None:
    visible = keypoints[keypoints[:, 2] > KEYPOINT_CONFIDENCE]
    if len(visible) < 2:
        return None

    x1, y1 = visible[:, :2].min(axis=0)
    x2, y2 = visible[:, :2].max(axis=0)
    pad_x = max((x2 - x1) * 0.25, 15)
    pad_y = max((y2 - y1) * 0.25, 15)
    return [
        max(0, float(x1 - pad_x)),
        max(0, float(y1 - pad_y)),
        min(FRAME_WIDTH, float(x2 + pad_x)),
        min(FRAME_HEIGHT, float(y2 + pad_y)),
    ]


def _iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection + 1e-6)


def _remove_duplicates(people: list[dict]) -> list[dict]:
    keep: list[dict] = []
    for person in sorted(people, key=lambda item: item["score"], reverse=True):
        if all(_iou(person["box"], other["box"]) <= 0.45 for other in keep):
            keep.append(person)
    return keep


def _camera_callback(request, imx500: IMX500) -> None:
    outputs = imx500.get_outputs(metadata=request.get_metadata(), add_batch=True)
    people = []

    if outputs is not None:
        keypoints_raw, scores, _ = postprocess_higherhrnet(
            outputs=outputs,
            img_size=(FRAME_HEIGHT, FRAME_WIDTH),
            img_w_pad=(0, 0),
            img_h_pad=(0, 0),
            detection_threshold=KEYPOINT_CONFIDENCE,
            network_postprocess=True,
        )
        if scores is not None and len(scores):
            keypoints_all = np.reshape(
                np.stack(keypoints_raw, axis=0), (len(scores), 17, 3)
            )
            for keypoints, score in zip(keypoints_all, scores):
                if score < PERSON_CONFIDENCE:
                    continue
                box = _body_box(keypoints)
                if box and box[2] - box[0] >= MIN_BOX_SIZE and box[3] - box[1] >= MIN_BOX_SIZE:
                    people.append({"box": box, "score": float(score)})

    with MappedArray(request, "main") as mapped:
        frame = mapped.array.copy()

    with state["lock"]:
        state["frame"] = frame
        state["people"] = _remove_duplicates(people)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())

    imx500 = IMX500(MODEL_PATH)
    camera = Picamera2(imx500.camera_num)
    camera.pre_callback = lambda request: _camera_callback(request, imx500)
    camera.configure(
        camera.create_video_configuration(
            main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
            controls={"FrameRate": 15},
        )
    )

    try:
        imx500.show_network_fw_progress_bar()
        camera.start()
        log.info("Camera running. Press Q or Esc to quit.")

        while not stopping.is_set():
            with state["lock"]:
                frame = state["frame"]
                people = list(state["people"])
                if frame is not None:
                    frame = frame.copy()

            if frame is None:
                time.sleep(0.01)
                continue

            for person in people:
                x1, y1, x2, y2 = (int(value) for value in person["box"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Person {person['score']:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            cv2.putText(
                frame,
                f"People detected: {len(people)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.imshow("Person detection - Q/Esc to quit", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        stopping.set()
        camera.stop()
        cv2.destroyAllWindows()
        log.info("Camera stopped.")


if __name__ == "__main__":
    main()