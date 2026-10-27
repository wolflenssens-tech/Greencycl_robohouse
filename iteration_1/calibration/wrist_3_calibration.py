"""One capture at a time using production model, depth filtering and StableTarget.
Show tag ID 1 as in launch.py. Space/R captures again; Esc closes.
The frozen box is the last sample; the displayed Wrist 3 angle is the stable result.
No robot connection is opened.
Test-only difference: angles follow the long axis before stability filtering.
"""

import time
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

from iteration_1.calculate import (
    ACCEPTED_AREA, DETECTION_CONFIDENCE, inside_area,
    StableTarget, smooth_depth, target_measurement, point,
)

WINDOW = 'Wrist 3 calibration - one target'


def long_axis_angle(box):
    angle = float(np.degrees(box[4]))
    if box[2] < box[3]:
        angle += 90.0
    return angle % 180.0


def main():
    model = YOLO(Path(__file__).resolve().parents[1] / 'best.pt', task='obb')
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    align = rs.align(rs.stream.color)
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_25h9))
    stable = StableTarget()
    origin = None
    smoothed_mm = None
    frozen = None
    capture_number = 0
    started = False
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    print('Looking for one target. Stop other camera scripts first.', flush=True)
    try:
        pipeline.start(config)
        started = True
        while True:
            # Continue consuming frames so a new capture uses a recent image.
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if frozen is None and color and depth:
                preview = np.asanyarray(color.get_data()).copy()
                depth_mm = np.asanyarray(depth.get_data()).astype(np.float32)
                depth_mm *= depth.get_units() * 1000.0
                smoothed_mm = smooth_depth(depth_mm, smoothed_mm)
                intrinsics = depth.profile.as_video_stream_profile().intrinsics
                tags, ids, _ = detector.detectMarkers(preview)
                if ids is not None:
                    matches = [tag for tag, tag_id in zip(tags, ids.reshape(-1)) if tag_id == 1]
                    if len(matches) == 1:
                        measured = point(matches[0].reshape(4, 2).mean(axis=0), depth, intrinsics)
                        if measured is not None and np.all(np.isfinite(measured)):
                            origin = measured
                result = model.predict(preview, conf=DETECTION_CONFIDENCE, verbose=False)[0]
                now = time.monotonic()
                candidates = []
                if result.obb is not None and origin is not None:
                    boxes = result.obb.xywhr.cpu().numpy()
                    for index, box in enumerate(boxes):
                        if inside_area([box[:2]]):
                            xyz = target_measurement(smoothed_mm, intrinsics, box[:2], origin)
                            if xyz is not None:
                                candidates.append((index, box[:2], xyz, long_axis_angle(box)))
                candidate = stable.select(candidates, now)
                captured = None
                if candidate is not None:
                    index, center, xyz, grasp_angle = candidate
                    captured = stable.update(index, center, xyz, now, grasp_angle)
                if captured is not None:
                    box = boxes[index]
                    raw_angle = float(np.degrees(box[4]))
                    angle_deg = float(captured[5])
                    corners = result.obb.xyxyxyxy.cpu().numpy()[index]
                    cv2.polylines(preview, [np.rint(corners).astype(np.int32)],
                                  True, (0, 255, 0), 2)
                    cv2.circle(preview, tuple(np.rint(box[:2]).astype(int)),
                               4, (0, 255, 0), -1)
                    # Show the exact stable angle being compared with the wrist.
                    direction = np.array([np.cos(np.radians(angle_deg)),
                                          np.sin(np.radians(angle_deg))])
                    half_length = max(25.0, float(max(box[2:4])) / 2)
                    ends = [tuple(np.rint(box[:2] + sign * half_length * direction).astype(int))
                            for sign in (-1, 1)]
                    cv2.line(preview, ends[0], ends[1], (255, 255, 0), 2)
                    capture_number += 1
                    print(f'Capture {capture_number}: final Wrist3 angle = '
                          f'{angle_deg:.3f} deg | latest raw angle={raw_angle:.3f} deg | width={box[2]:.1f} px, '
                          f'height={box[3]:.1f} px', flush=True)
                    cv2.putText(preview, f'Long-axis angle: {angle_deg:.3f} deg',
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                    cv2.putText(preview, 'FROZEN | Space/R: next | Esc: close',
                                (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    frozen = preview.copy()
                else:
                    left, top, right, bottom = ACCEPTED_AREA
                    cv2.rectangle(preview, (left, top), (right, bottom), (0, 255, 255), 2)
                    status = 'Show ID 1 once with valid depth' if origin is None else stable.status
                    cv2.putText(preview, status,
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.imshow(WINDOW, preview)
            elif frozen is not None:
                cv2.imshow(WINDOW, frozen)

            key = cv2.waitKey(1) & 0xff
            if key == 27 or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (32, ord('r'), ord('R')) and frozen is not None:
                frozen = None
                stable.reset()
                smoothed_mm = None
                print('Looking for one new target...', flush=True)
    finally:
        if started:
            pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
