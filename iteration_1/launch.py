import sys
from pathlib import Path

# Support the editor's Run Python File button as well as module execution.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import threading

from iteration_1.communication_UR import PC_IP, PORT, ROBOT_IP, TargetExchange, create_server



def main():
    try:
        from camera_to_robot import camera_to_robot
    except ModuleNotFoundError as exc:
        raise SystemExit('Run calibration.py first to create calibration/camera_to_robot.py.') from exc
    from iteration_1.calculate import main as detect

    def transform(xyz, angle):
        x, y = camera_to_robot(xyz[0], xyz[1])
        return (x, y, xyz[2]), angle

    exchange = TargetExchange()
    with create_server(PC_IP, PORT, ROBOT_IP, exchange) as server:
        rpc = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.1}, daemon=True)
        rpc.start()
        exchange.start()
        try:
            detect(exchange, transform)
        finally:
            exchange.disconnect()
            server.shutdown()
            rpc.join()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass

