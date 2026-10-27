import json
import threading
import time
from pathlib import Path
from xmlrpc.server import SimpleXMLRPCRequestHandler, SimpleXMLRPCServer
from socketserver import ThreadingMixIn

import cv2
import numpy as np
import pyrealsense2 as rs

PC_IP = '192.168.0.20'
PORT = 50000
POSITIONS = 20
CAPTURE_SECONDS = 5.0


class Exchange:
    def __init__(self):
        self.condition = threading.Condition()
        self.feedback = None
        self.busy = False
        self.finished = False
        self.go = None

    def send_xy(self, x, y):
        xy = np.array([x, y], dtype=float)
        with self.condition:
            if self.finished:
                return False
            if self.busy:
                raise ValueError('Already capturing a position.')
            if xy.shape != (2,) or not np.all(np.isfinite(xy)):
                raise ValueError('Invalid robot XY.')
            self.busy = True
            self.go = None
            self.feedback = xy * 1000
            # This RPC reply is the GO message; the UR waits here during capture.
            self.condition.wait_for(lambda: self.go is not None or self.finished)
            self.busy = False
            return bool(self.go) and not self.finished

    def take_feedback(self):
        with self.condition:
            xy = self.feedback
            self.feedback = None
            return xy

    def release(self):
        with self.condition:
            self.go = True
            self.condition.notify_all()

    def finish(self):
        with self.condition:
            self.finished = True
            self.condition.notify_all()


class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/', '/RPC2')

class RPCServer(ThreadingMixIn, SimpleXMLRPCServer):
    pass


def point(center, depth_frame, intrinsics):
    u, v = map(int, center)
    depth = np.asanyarray(depth_frame.get_data())
    patch = depth[max(0, v - 2):v + 3, max(0, u - 2):u + 3]
    valid = patch[patch > 0]
    if not valid.size:
        return None
    distance = float(np.median(valid)) * depth_frame.get_units()
    return np.array(rs.rs2_deproject_pixel_to_point(intrinsics, [float(u), float(v)], distance)) * 1000


def fit_transform(camera, robot):
    """Fit rotation, uniform scale and translation; inputs/outputs are mm."""
    camera = np.asarray(camera, dtype=float)
    robot = np.asarray(robot, dtype=float)
    if (camera.ndim != 2 or camera.shape[1] != 2 or
            robot.shape != camera.shape or len(camera) < 3 or
            not np.all(np.isfinite(camera)) or not np.all(np.isfinite(robot))):
        raise ValueError('Provide at least three finite camera/robot XY pairs.')
    if np.linalg.matrix_rank(camera - camera.mean(axis=0)) < 2:
        raise ValueError('Spread calibration points across both X and Y.')
    x, y = camera.T
    design = np.zeros((2 * len(camera), 4))
    design[0::2] = np.column_stack((x, y, np.ones(len(camera)), np.zeros(len(camera))))
    design[1::2] = np.column_stack((y, -x, np.zeros(len(camera)), np.ones(len(camera))))
    a, b, tx, ty = np.linalg.lstsq(design, robot.ravel(), rcond=None)[0]
    coefficients = np.array([[a, b, tx], [-b, a, ty]])
    errors = np.column_stack((camera, np.ones(len(camera)))) @ coefficients.T - robot
    return coefficients, errors


def save_result(camera, robot):
    coefficients, errors = fit_transform(camera, robot)
    root = Path(__file__).resolve().parent
    expressions = [f'{a!r}*x + {b!r}*y + {t!r}'
                   for a, b, t in coefficients.tolist()]
    function = 'def camera_to_robot(x, y):\n    return (' + ', '.join(expressions) + ')\n'
    (root / 'camera_to_robot.py').write_text(function, encoding='utf-8')
    distances = np.linalg.norm(errors, axis=1)
    result = {
        'units': 'mm',
        'camera_coordinates': 'offset = ID1 point minus ID2 point; camera_x = offset_y; camera_y = offset_x; aligned depth',
        'model': 'robot_x = a*x + b*y + tx; robot_y = -b*x + a*y + ty',
        'coefficient_order': ['camera_x', 'camera_y', 'translation'],
        'coefficients': {'x': coefficients[0].tolist(), 'y': coefficients[1].tolist()},
        'camera_xy': np.asarray(camera).tolist(),
        'robot_xy': np.asarray(robot).tolist(),
        'fit_errors_xy': errors.tolist(),
        'fit_rms_mm': float(np.sqrt(np.mean(distances**2))),
        'fit_max_mm': float(np.max(distances)),
    }
    (root / 'calibration.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


def main():
    exchange = Exchange()
    server = RPCServer((PC_IP, PORT), requestHandler=RequestHandler,
                       logRequests=False)
    server.register_function(exchange.send_xy)
    rpc = threading.Thread(target=server.serve_forever,
                           kwargs={'poll_interval': 0.1}, daemon=True)
    pipeline = rs.pipeline()
    started = False
    rpc.start()
    camera, robot = [], []
    try:
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipeline.start(config)
        started = True
        align = rs.align(rs.stream.color)
        detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_25h9))
        pending, samples = None, []
        while len(camera) < POSITIONS:
            if pending is None:
                pending = exchange.take_feedback()
                if pending is not None:
                    # Ignore buffered frames from before the robot stopped.
                    capture_after = time.monotonic() + 0.2
                    deadline = capture_after + CAPTURE_SECONDS
                    samples = []
            frames = align.process(pipeline.wait_for_frames())
            color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
            now = time.monotonic()
            rgb = None
            measured = None
            if color_frame and depth_frame:
                rgb = np.asanyarray(color_frame.get_data()).copy()
                if pending is not None and now >= capture_after:
                    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
                    corners, ids, _ = detector.detectMarkers(rgb)
                    centers = {}
                    if ids is not None:
                        for tag, tag_id in zip(corners, ids.reshape(-1)):
                            center = tuple(tag.reshape(4, 2).mean(axis=0).astype(int))
                            centers.setdefault(int(tag_id), []).append(center)
                            cv2.circle(rgb, center, 5, (0, 0, 255), -1)
                    if len(centers.get(1, [])) == 1 and len(centers.get(2, [])) == 1:
                        base = point(centers[1][0], depth_frame, intrinsics)
                        tip = point(centers[2][0], depth_frame, intrinsics)
                        if base is not None and tip is not None:
                            offset = base - tip
                            measured = np.array([offset[1], offset[0]])
                            if not np.all(np.isfinite(measured)):
                                measured = None
            if pending is not None and now >= capture_after:
                if measured is None:
                    samples = []
                else:
                    samples.append(measured)
                    samples = samples[-30:]
                captured = (len(samples) == 30 and
                            np.max(np.ptp(samples, axis=0)) <= 6)
                if captured:
                    camera.append(np.median(samples, axis=0))
                    robot.append(pending)
                if captured or now >= deadline:
                    pending = None
                    if len(camera) < POSITIONS:
                        exchange.release()
            if rgb is not None:
                status = 'Recording tags' if pending is not None else 'Waiting for UR'
                cv2.putText(rgb, f'{len(camera)}/{POSITIONS} | {status} | Esc: exit',
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow('RGB calibration', rgb)
            if cv2.waitKey(1) & 0xff == 27:
                break
        if len(camera) == POSITIONS:
            save_result(camera, robot)
    finally:
        exchange.finish()
        if started:
            pipeline.stop()
        cv2.destroyAllWindows()
        server.shutdown()
        server.server_close()
        rpc.join()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
