from pathlib import Path
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from iteration_1.calibration.xy_calibration import point

ACCEPTED_AREA = (65, 10, 620, 370)
BASE_TAG_ID = 1
BASE_EXCLUSION_RADIUS_MM = 250.0
STABLE_SECONDS = 0.6
MIN_STABLE_SAMPLES = 4
MAX_SAMPLE_GAP = 1.0
MATCH_RADIUS_PX = 35.0
PIXEL_TOLERANCE = 40.0
DETECTION_CONFIDENCE = 0.15
XYZ_TOLERANCE_MM = 50.0
REFERENCE_DEPTH_MM = 1337.0 - 57.5
SMOOTH_ALPHA = 0.15
RESET_JUMP_MM = 30.0
GRASP_DEPTH_OFFSET = 0.65  
MIN_GRASP_HEIGHT_MM = 10.0
REFERENCE_PLANE_ROBOT_Z_MM = 0.0

target_xyz_mm = None
target_pixel = None
target_track_id = None
target_rotation_deg = None


def smooth_depth(current, previous):
    if previous is None:
        return current.copy()
    result = current.copy()
    blend = ((current > 0) & (previous > 0) &
             (np.abs(current - previous) <= RESET_JUMP_MM))
    result[blend] = previous[blend] + SMOOTH_ALPHA * (current[blend] - previous[blend])
    return result


def inside_area(corners):
    left, top, right, bottom = ACCEPTED_AREA
    corners = np.asarray(corners)
    return bool(np.all(np.isfinite(corners)) and
                np.all((corners[:, 0] >= left) & (corners[:, 0] <= right) &
                       (corners[:, 1] >= top) & (corners[:, 1] <= bottom)))


def outside_base_zone(xyz, origin):
    """Use camera-plane XY in mm; height does not affect the exclusion circle."""
    if xyz is None or origin is None:
        return False
    xy = np.asarray(xyz, dtype=float)[:2]
    base_xy = np.asarray(origin, dtype=float)[:2]
    return bool(np.all(np.isfinite(xy)) and np.all(np.isfinite(base_xy)) and
                np.linalg.norm(xy - base_xy) > BASE_EXCLUSION_RADIUS_MM)


def update_base_origin(corners, ids, depth_frame, intrinsics, origin):
    if ids is not None:
        matches = [tag for tag, tag_id in zip(corners, ids.reshape(-1))
                   if tag_id == BASE_TAG_ID]
        if len(matches) == 1:
            measured = point(matches[0].reshape(4, 2).mean(axis=0), depth_frame, intrinsics)
            if measured is not None and np.all(np.isfinite(measured)) and measured[2] > 0:
                return measured
    return origin


def target_measurement(depth_mm, intrinsics, center, origin=None):
    u, v = map(int, np.rint(center))
    h, w = depth_mm.shape
    if not (2 <= u < w - 2 and 2 <= v < h - 2):
        return None
    patch = depth_mm[v - 2:v + 3, u - 2:u + 3]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size < 5:
        return None
    distance = float(np.median(valid))
    height = REFERENCE_DEPTH_MM - distance
    if height < MIN_GRASP_HEIGHT_MM:
        return None
    x, y, _ = rs.rs2_deproject_pixel_to_point(
        intrinsics, [float(u), float(v)], distance / 1000.0)
    x, y = x * 1000.0, y * 1000.0
    if origin is not None:
        x, y = origin[1] - y, origin[0] - x
    grasp_height_mm = max(MIN_GRASP_HEIGHT_MM, height * (1.0 - GRASP_DEPTH_OFFSET))
    robot_z_mm = REFERENCE_PLANE_ROBOT_Z_MM + grasp_height_mm
    return np.array([x, y, robot_z_mm])


class StableTarget:
    def __init__(self):
        self.next_id = 1
        self.reset()

    def reset(self):
        self.track_id = None
        self.anchor = None
        self.samples = []
        self.saved = False
        self.status = 'Waiting for target'

    def missing(self, now):
        if self.samples and now - self.samples[-1][0] > MAX_SAMPLE_GAP:
            self.reset()

    def select(self, candidates, now):
        self.missing(now)
        if self.anchor is None:
            return choose_target(candidates) if candidates else None
        nearby = [c for c in candidates if
                  np.linalg.norm(c[1] - self.anchor) <= MATCH_RADIUS_PX]
        return min(nearby, key=lambda c: np.linalg.norm(c[1] - self.anchor)) if nearby else None

    def update(self, track_id, pixel, xyz, now, angle=0.0):
        if xyz is None or not np.all(np.isfinite(xyz)):
            self.missing(now)
            return None
        sample = np.r_[pixel, xyz, angle].astype(float)
        if (not self.samples or now - self.samples[-1][0] > MAX_SAMPLE_GAP or
                np.linalg.norm(np.asarray(pixel) - self.anchor) > MATCH_RADIUS_PX):
            self.reset()
            self.track_id = self.next_id
            self.next_id += 1
            self.anchor = np.asarray(pixel, dtype=float).copy()
        if self.saved:
            self.samples = [(now, sample)]
            return None
        self.status = 'Gathering stable samples'
        self.samples.append((now, sample))
        while len(self.samples) > MIN_STABLE_SAMPLES and now - self.samples[1][0] >= STABLE_SECONDS:
            self.samples.pop(0)
        if (now - self.samples[0][0] < STABLE_SECONDS or
                len(self.samples) < MIN_STABLE_SAMPLES):
            return None
        points = np.array([s[1] for s in self.samples])
        spread = np.percentile(points, 90, axis=0) - np.percentile(points, 10, axis=0)
        if np.any(spread[:2] > PIXEL_TOLERANCE):
            self.status = 'Target position is moving'
            return None
        if np.any(spread[2:5] > XYZ_TOLERANCE_MM):
            self.status = 'Target depth / XYZ is unstable'
            return None
        angles = points[:, 5]
        delta = (angles - angles[0] + 90.0) % 180.0 - 90.0
        if np.percentile(delta, 90) - np.percentile(delta, 10) > 25.0:
            self.status = 'Bounding-box angle is unstable'
            return None
        captured = np.median(points, axis=0)
        captured[5] = (angles[0] + np.median(delta)) % 180.0
        self.saved = True
        self.status = 'Target captured'
        return captured


def choose_target(candidates):
    left, top, right, bottom = ACCEPTED_AREA
    middle = np.array([(left + right) / 2, (top + bottom) / 2])
    return min(candidates, key=lambda c: (float(np.linalg.norm(c[1] - middle)), c[0]))


def main(exchange=None, transform=None):
    global target_xyz_mm, target_pixel, target_track_id, target_rotation_deg
    model = YOLO(Path(__file__).with_name('best.pt'), task='obb')
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    align = rs.align(rs.stream.color)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_25h9)
    detector = cv2.aruco.ArucoDetector(dictionary)
    stable = StableTarget()
    active_request = None
    frozen_preview = None
    origin = None
    smoothed_mm = None
    pipeline.start(config)
    try:
        while True:
            request = exchange.capture_request() if exchange is not None else None
            if exchange is not None and request != active_request:
                stable.reset()
                active_request = request
                smoothed_mm = None
                if request is not None:
                    frozen_preview = None
                    target_xyz_mm = target_pixel = target_track_id = target_rotation_deg = None
            frames = pipeline.wait_for_frames()
            # Retain the captured image through idle/robot movement until a new request.
            if frozen_preview is not None:
                cv2.imshow('RGB', frozen_preview)
                key = cv2.waitKey(1) & 0xff
                if key == 27:
                    break
                if exchange is None and key in (ord('r'), ord('c')):
                    frozen_preview = None
                    stable.reset()
                    smoothed_mm = None
                    target_xyz_mm = target_pixel = target_track_id = target_rotation_deg = None
                continue
            captured_this_frame = False
            # Locate the base even before a capture request.
            # Object inference still only runs during capture.
            if exchange is not None and active_request is None:
                frames = align.process(frames)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if color_frame:
                    rgb = np.asanyarray(color_frame.get_data()).copy()
                    if depth_frame:
                        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
                        corners, ids, _ = detector.detectMarkers(rgb)
                        origin = update_base_origin(corners, ids, depth_frame, intrinsics, origin)
                    cv2.putText(rgb, exchange.status(), (10, rgb.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.imshow('RGB', rgb)
                if cv2.waitKey(1) & 0xff == 27:
                    break
                continue
            frames = align.process(frames)
            color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
            if not color_frame or not depth_frame:
                stable.missing(time.monotonic())
                continue
            rgb = np.asanyarray(color_frame.get_data()).copy()
            depth_mm = np.asanyarray(depth_frame.get_data()).astype(np.float32)
            depth_mm *= depth_frame.get_units() * 1000.0
            smoothed_mm = smooth_depth(depth_mm, smoothed_mm)
            depth_mm = smoothed_mm
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            corners, ids, _ = detector.detectMarkers(rgb)
            result = model.predict(rgb, conf=DETECTION_CONFIDENCE, verbose=False)[0]
            rgb = result.plot()
            now = time.monotonic()
            origin = update_base_origin(corners, ids, depth_frame, intrinsics, origin)
            capturing = origin is not None and (exchange is None or active_request is not None)
            candidates = []
            inside_count = 0
            excluded_count = 0
            obb = result.obb
            if obb is not None:
                for track_id, box in enumerate(obb.xywhr.cpu().numpy()):
                    center = box[:2]
                    in_rectangle = inside_area([center])
                    inside_count += int(in_rectangle)
                    # Measure in camera coordinates to compare directly with the base tag.
                    xyz = target_measurement(depth_mm, intrinsics, center)
                    outside_base = outside_base_zone(xyz, origin)
                    excluded_count += int(in_rectangle and xyz is not None and
                                          origin is not None and not outside_base)
                    accepted = in_rectangle and outside_base
                    cv2.circle(rgb, tuple(np.rint(center).astype(int)), 5,
                               (0, 255, 0) if accepted else (0, 0, 255), -1)
                    if not accepted:
                        box_corners = obb.xyxyxyxy[track_id].cpu().numpy()
                        cv2.polylines(rgb, [np.rint(box_corners).astype(np.int32)],
                                      True, (0, 0, 255), 2)
                    if accepted and capturing:
                        if exchange is not None:
                            xyz = np.array([origin[1] - xyz[1], origin[0] - xyz[0], xyz[2]])
                        if xyz is not None:
                            angle = float(np.degrees(box[4]))
                            if box[2] < box[3]:
                                angle += 90.0
                            angle %= 180.0
                            candidates.append((int(track_id), center, xyz, angle))
            candidate = stable.select(candidates, now)
            if candidate is not None:
                track_id, center, xyz, angle = candidate
                captured = stable.update(track_id, center, xyz, now, angle)
                if captured is not None:
                    xyz = tuple(map(float, captured[2:5]))
                    angle = float(captured[5])
                    if exchange is not None:
                        robot_xyz, wrist_angle = transform(xyz, angle)
                        if not exchange.set_target(active_request, robot_xyz, wrist_angle):
                            continue
                    target_pixel = tuple(map(float, captured[:2]))
                    target_xyz_mm = xyz
                    target_rotation_deg = angle
                    target_track_id = stable.track_id
                    captured_this_frame = True

            if captured_this_frame:
                status = ('Target found: waiting for next request' if exchange is not None
                          else 'Target found: waiting for next request')
            elif exchange is not None and active_request is None:
                status = exchange.status()
            elif origin is None:
                status = f'Show ID {BASE_TAG_ID} once with valid depth'
            elif not inside_count:
                status = 'No target centre inside the area'
            elif not candidates and excluded_count:
                status = f'Target inside base exclusion zone ({BASE_EXCLUSION_RADIUS_MM:g} mm)'
            elif not candidates:
                status = f'Target depth missing or height below {MIN_GRASP_HEIGHT_MM:g} mm'
            elif candidate is None:
                status = 'Waiting for selected target'
            else:
                status = stable.status
            cv2.putText(rgb, status, (10, rgb.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if ids is not None:
                for tag, tag_id in zip(corners, ids.reshape(-1)):
                    center = tuple(tag.reshape(4, 2).mean(axis=0).astype(int))
                    cv2.circle(rgb, center, 5, (0, 0, 255), -1)
                    cv2.putText(rgb, f'ID {tag_id}', (center[0] + 8, center[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            left, top, right, bottom = ACCEPTED_AREA
            cv2.rectangle(rgb, (left, top), (right, bottom), (0, 255, 255), 2)
            if captured_this_frame:
                frozen_preview = rgb.copy()
            cv2.imshow('RGB', rgb)
            key = cv2.waitKey(1) & 0xff
            if key == 27:
                break
            if key in (ord('r'), ord('c')) and frozen_preview is None:
                stable.reset()
                target_xyz_mm = target_pixel = target_track_id = target_rotation_deg = None
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
