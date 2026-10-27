# Greencycl automation

Object detection and grasp-target generation for a Universal Robots arm using an Intel RealSense depth camera and a trained YOLO model.

The system detects objects, estimates their position and orientation, converts camera measurements into robot coordinates, and sends one target at a time to the robot over XML-RPC.

## Workflow — Python side

1. Capture aligned colour and depth images from the RealSense camera.
2. Locate AprilTag ID 1 as the base reference.
3. Detect objects using the YOLO oriented bounding box model.
4. Filter object centres by the working area and the exclusion zone around the base reference.
5. Select a target and wait for stable position, depth, and angle measurements.
6. Calculate robot XY coordinates, grasp height, and Wrist 3 angle.
7. Send the target to the robot and wait for its completion signal before capturing again.

## Workflow — UR5e side (Local mode)

The actual pendant program follows this sequence:

```text
Initialize gripper
MoveJ to Home
Open gripper
Loop
    Script: UR5e_target.script
    Close gripper
    MoveJ to safe point
    MoveJ to drop point
    Open gripper
    Script: UR5e_send_bit.script
    MoveJ to safe point
```

Home is used during initialization. The safe point is a taught position in the air used between the target and drop point, and again before the next loop iteration, to route the arm around surrounding objects.

The completion signal is sent after opening the gripper at the drop point and before the final MoveJ to the safe point. This starts the next camera capture while the robot returns to the safe point. On the next iteration, `UR5e_target.script` waits if the new target is not yet ready.

## Folder contents

| File | Description |
| --- | --- |
| `launch.py` | Starts the detection pipeline and robot communication server. |
| `calculate.py` | Handles detection, depth filtering, target selection, stability checks, and grasp estimation. |
| `communication_UR.py` | Exchanges targets, feedback, and completion signals with the robot over XML-RPC. |
| `camera_to_robot.py` | Contains the calibrated transformation from camera XY to robot XY. |
| `best.pt` | Trained YOLO oriented bounding box model. |
| `UR5e_target.script` | Requests a target, solves the joint configuration, executes `movej`, and reports the measured pose. |
| `UR5e_send_bit.script` | Sends the completion signal that requests the next camera capture. |
| `calibration/UR5e_random_target.script` | Generates a reachable calibration position and moves the robot there. |
| `calibration/UR5e_send_XY.script` | Sends the measured robot XY position and waits for the PC to finish capturing. |
| `calibration/xy_calibration.py` | Calculates the XY transformation using AprilTags and robot position feedback. |
| `calibration/calibration.json` | Stores calibration measurements, transformation coefficients, and fitting errors. |
| `calibration/wrist_3_calibration.py` | Displays stable object angles for checking wrist alignment without a robot connection. |
| `calibration/depth_viewer.py` | Displays depth measurements and estimated heights for inspection. |

## Requirements

- Intel RealSense depth camera.
- Universal Robots arm and a corresponding robot-side program.
- Python with NumPy, OpenCV with `cv2.aruco` support, `pyrealsense2`, and Ultralytics.
- AprilTags from the `DICT_APRILTAG_25h9` dictionary: ID 1 for the base reference and ID 2 for XY calibration.

## Running

From the parent directory of `iteration_1`, run:

```powershell
python iteration_1/launch.py
```

Keep the folder named `iteration_1`, as the code uses this name in its imports.

Default network settings in `communication_UR.py`:

| Setting | Value |
| --- | --- |
| PC IP | `192.168.0.20` |
| Robot IP | `192.168.0.10` |
| Port | `50000` |

The server accepts connections only from the configured robot IP. Show AprilTag ID 1 with valid depth before capturing a target. Press **Esc** to stop.

## Robot communication

The XML-RPC server exposes three methods:

- `recv_target()` returns `[x, y, z, wrist3]` in **metres** and **radians**. Each target is sent once. If no new target is available, the server returns `[10000.0, 0.0, 0.0, 0.0]` as an idle marker.
- `send_data(current_pose, actual_wrist3_degrees)` checks robot feedback against the target, with tolerances of 1 mm for position and 0.5 degrees for Wrist 3. The feedback wrist angle is supplied in degrees.
- `receive_bit(1)` signals completion of an active target cycle and starts the next capture. Sending feedback alone does not complete the cycle.

## UR5e local-mode program

The UR5e runs in **Local mode**. The operator starts the program from the teach pendant. Python runs on the PC, while the robot's local program combines the supplied Script nodes with the setup's MoveJ, Home, and gripper nodes.

The `.script` files are included here; the complete pendant program and its taught Home/waypoint configuration are not included.

The exact node order is shown in **Workflow — UR5e side (Local mode)** above. The Script nodes handle target acquisition and completion signalling; the gripper and taught MoveJ nodes carry out the handling sequence.

`UR5e_target.script` polls `recv_target()` until a target is available. It uses the active TCP and inverse kinematics to find a joint configuration matching the requested XYZ and absolute Wrist 3 angle, with TCP +Z aligned with base +Z at the endpoint. It then executes `movej(gc_q, a=0.4, v=0.4, r=0)`, checks the measured endpoint, and reports feedback to the PC. The target script therefore already contains the move to the detected target; the pendant's other MoveJ and Home nodes handle the rest of the sequence.

`UR5e_send_bit.script` calls `receive_bit(1)`. Keep this separate from target acquisition so the local robot program controls when the next cycle begins.

Start `launch.py` on the PC before starting the local robot program. The scripts use `http://192.168.0.20:50000`; if the PC address or port changes, update both the Python configuration and the script URLs.

**Current implementation detail:** the target solver can search for an adjusted XYZ up to 25 mm along one axis if the original target cannot be solved. Python still validates feedback against the original target with a 1 mm tolerance, so a larger accepted adjustment can be rejected during feedback. The script's failure message says 20 mm, but its configured maximum is 25 mm.

## Calibration tools

Run these from the parent directory of `iteration_1`:

```powershell
python iteration_1/calibration/xy_calibration.py
python iteration_1/calibration/wrist_3_calibration.py
python iteration_1/calibration/depth_viewer.py
```

### XY calibration

XY calibration pairs camera measurements with actual robot TCP positions. It fits rotation, uniform scale, and translation in the XY plane; it does not calibrate Z or Wrist 3.

1. Keep AprilTag ID 1 fixed as the base reference. Arrange ID 2 so its measured XY corresponds to the robot TCP position being reported. Both tags must be visible with valid depth during capture.
2. Stop `launch.py` before starting calibration: both programs use the camera and port `50000`.
3. Run `python iteration_1/calibration/xy_calibration.py` on the PC.
4. In the UR5e's local calibration program, repeat these Script nodes in order:

   ```text
   Loop
       UR5e_random_target.script
       UR5e_send_XY.script
   ```

5. Start the calibration program from the teach pendant. The first script selects a reachable position and executes MoveJ. The second reads the actual TCP XY, sends it in metres through `send_xy(x, y)`, and blocks while the PC captures the markers.
6. Let the PC collect 20 accepted positions. For each position, it waits 0.2 seconds before sampling, requires 30 consecutive valid samples with no more than 6 mm spread per XY axis, and records their median. A capture that times out after 5 seconds is skipped, and the robot is released to try another position.
7. After 20 accepted positions, the PC saves the calibration and returns `False` to the waiting robot script, which halts the calibration program.

The random-target script currently samples base-frame X from -690 to -240 mm and Y from -640 to 640 mm, at Z = 350 mm. These are setup-specific calibration bounds defined in `UR5e_random_target.script`.

Camera coordinates are calculated from the difference between the ID 1 and ID 2 positions, with camera X taken from the Y difference and camera Y from the X difference. The fitted mapping is:

```text
robot_x = a * camera_x + b * camera_y + tx
robot_y = -b * camera_x + a * camera_y + ty
```

The script writes these files inside `calibration/`:

- `camera_to_robot.py`: the generated transformation function.
- `calibration.json`: paired measurements, coefficients, individual fitting errors, RMS error, and maximum error, in millimetres.

**Apply the new calibration:** copy the generated `calibration/camera_to_robot.py` to `iteration_1/camera_to_robot.py`, then restart the launcher. The launcher currently reads the root-level copy. Exiting before all 20 positions are collected does not save a new fit.

Repeat XY calibration when the camera, base reference, or robot-to-camera arrangement changes. Check grasp height and wrist alignment separately using the tools below.

### Wrist angle inspection

Displays the detected object's stable long-axis angle. Press **Space** or **R** to capture another target, and **Esc** to close.

### Depth inspection

Displays the camera image and depth heatmap. Click the depth image to inspect a pixel. Use **F** to focus the displayed depth range, **W** to widen it, **R** to reset, and **Q** or **Esc** to close.

## Main configuration

Settings are defined at the top of `calculate.py`.

| Parameter | Default |
| --- | --- |
| Accepted image region | `(65, 10, 620, 370)` pixels |
| Detection confidence | `0.15` |
| Base exclusion radius | `250 mm` |
| Stability window | At least `0.6 seconds` and `4 samples` |
| Reference depth | `1279.5 mm` |
| Minimum object height | `10 mm` |
| Grasp height | 35% of estimated object height, with a 10 mm minimum |

Calibration values and reference depth are specific to the current camera and robot arrangement.


