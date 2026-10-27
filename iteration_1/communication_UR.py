import argparse
import math
import threading
from socketserver import ThreadingMixIn
from xmlrpc.server import SimpleXMLRPCRequestHandler, SimpleXMLRPCServer

PC_IP = "192.168.0.20"
ROBOT_IP = "192.168.0.10"
PORT = 50000

# TARGET_XYZ_MM = (-630.0, -364, 86.0)
# TARGET_WRIST3_DEG = 55.045
TARGET_XYZ_MM = None
TARGET_WRIST3_DEG = None


def configured_target(xyz=None, angle=None):
    if xyz is None:
        xyz, angle = TARGET_XYZ_MM, TARGET_WRIST3_DEG
    if xyz is None:
        return None
    if len(xyz) != 3:
        raise ValueError("TARGET_XYZ_MM must contain X, Y and Z.")
    values = [float(v) / 1000.0 for v in xyz]
    values.append(math.radians(float(angle)))
    if not all(math.isfinite(v) and abs(v) < 1e100 for v in values):
        raise ValueError("Target must contain four finite, representable numbers.")
    if values[0] == 10000.0:
        raise ValueError("X=10000 metres is reserved for idle.")
    if abs(values[3]) > 2 * math.pi:
        raise ValueError("UR5e absolute Wrist 3 must be within -360 to +360 degrees.")
    return values


class TargetExchange:
    def __init__(self):
        self.target = None
        self.generation = 0
        self.request = None
        self.state = None
        self.lock = threading.Lock()
        self.pending = False
        self.in_flight = False
        self.last_feedback = None

    def start(self):
        with self.lock:
            self.state = "RUNNING"
            self._request_next()

    def _request_next(self):
        # Caller holds the lock. A new generation discards old camera samples.
        self.generation += 1
        self.request = self.generation
        self.target = None
        self.pending = self.in_flight = False
        self.last_feedback = None

    def receive_bit(self, value):
        if type(value) is not int or value not in (0, 1):
            raise ValueError("Expected integer 0 or 1")
        with self.lock:
            if value == 1 and self.state == "RUNNING" and self.in_flight:
                self._request_next()
        return True

    def status(self):
        with self.lock:
            if self.state != "RUNNING":
                return "Stopped"
            if self.pending:
                return "Target ready; waiting for UR to request it"
            if self.in_flight:
                return "Waiting for UR completion message: 1"
            return "Finding a new target"

    def capture_request(self):
        with self.lock:
            return self.request if self.state == "RUNNING" else None

    def set_target(self, request, xyz, angle):
        global TARGET_XYZ_MM, TARGET_WRIST3_DEG
        target = configured_target(xyz, angle)
        with self.lock:
            if request is None or request != self.request or self.state != "RUNNING":
                return False
            TARGET_XYZ_MM, TARGET_WRIST3_DEG = tuple(xyz), float(angle)
            self.target = target
            self.request = None
            self.pending = True
            return True

    def disconnect(self):
        with self.lock:
            self.state = None
            self.request = None
            self.target = None
            self.pending = self.in_flight = False

    def recv_target(self):
        with self.lock:
            if self.state == "RUNNING" and self.pending and not self.in_flight:
                self.pending = False
                self.in_flight = True
                x, y, z = (value * 1000.0 for value in self.target[:3])
                angle = math.degrees(self.target[3])
                print(f'Sent to UR: X={x:.3f} mm, Y={y:.3f} mm, '
                      f'Z={z:.3f} mm, Wrist3={angle:.3f} deg', flush=True)
                return list(self.target)
        return [10000.0, 0.0, 0.0, 0.0]

    def send_data(self, current_pose, actual_wrist3_degrees):
        keys = ("x", "y", "z", "rx", "ry", "rz")
        if isinstance(current_pose, dict):
            pose = [float(current_pose[key]) for key in keys]
        else:
            pose = [float(v) for v in current_pose]
        angle = math.radians(float(actual_wrist3_degrees))
        if len(pose) != 6 or not all(math.isfinite(v) for v in pose + [angle]):
            raise ValueError("Invalid robot feedback.")
        with self.lock:
            if not self.in_flight:
                raise ValueError("No target is awaiting feedback.")
            position_error = math.sqrt(sum((pose[i] - self.target[i]) ** 2 for i in range(3)))
            if position_error > 0.001 or abs(angle - self.target[3]) > math.radians(0.5):
                raise ValueError("Robot feedback differs from target XYZ or Wrist 3.")
            self.last_feedback = (pose, actual_wrist3_degrees)
            # Only receive_bit(1) completes this cycle.
        return True


class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ("/", "/RPC2")
    protocol_version = "HTTP/1.1"


class RPCServer(ThreadingMixIn, SimpleXMLRPCServer):
    daemon_threads = True
    block_on_close = False
    allow_reuse_address = False

    def verify_request(self, request, address):
        return address[0] == self.allowed_robot_ip


def create_server(host, port, robot_ip, exchange):
    server = RPCServer((host, port), requestHandler=RequestHandler,
                       allow_none=False, logRequests=False)
    server.allowed_robot_ip = robot_ip
    server.register_function(exchange.recv_target, "recv_target")
    server.register_function(exchange.send_data, "send_data")
    server.register_function(exchange.receive_bit, "receive_bit")
    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=PC_IP)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--robot-ip", default=ROBOT_IP)
    parser.add_argument("--check", action="store_true",
                        help="Validate the configured target and exit without opening any network connection")
    args = parser.parse_args()
    configured_target()
    if args.check:
        return
    exchange = TargetExchange()
    with create_server(args.host, args.port, args.robot_ip, exchange) as server:
        exchange.start()
        try:
            server.serve_forever(poll_interval=0.1)
        finally:
            exchange.disconnect()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, TypeError) as exc:
        raise SystemExit(f"UR: {exc}") from exc
