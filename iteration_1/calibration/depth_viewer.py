import cv2
import numpy as np
import pyrealsense2 as rs

MAX_DEPTH_MM = 1337.0 - 57.5
MIN_MM = 700
MAX_MM = 1500
WINDOW = 'RealSense depth debugger'
SMOOTH_ALPHA = 0.15 
RESET_JUMP_MM = 30.0  


def smooth_depth(current, previous):
    if previous is None:
        return current.copy()
    result = current.copy()
    blend = ((current > 0) & (previous > 0) &
             (np.abs(current - previous) <= RESET_JUMP_MM))
    result[blend] = previous[blend] + SMOOTH_ALPHA * (current[blend] - previous[blend])
    return result


def colourise(depth_mm):
    scaled = np.clip((depth_mm - MIN_MM) / (MAX_MM - MIN_MM), 0, 1)
    colours = cv2.applyColorMap(((1 - scaled) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    colours[depth_mm <= 0] = 0
    return colours


def label(image, text, x, y):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (235, 235, 235), 1, cv2.LINE_AA)


def main():
    global MIN_MM, MAX_MM
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    selected = [320, 240]
    started = False
    smoothed_mm = None

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and 640 <= x < 1280 and 0 <= y < 480:
            selected[:] = [x - 640, y]

    try:
        pipeline.start(config)
        started = True
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, click)
        while True:
            frames = pipeline.wait_for_frames(5000)
            depth, color = frames.get_depth_frame(), frames.get_color_frame()
            if not depth or not color:
                continue
            measured_mm = np.asanyarray(depth.get_data()).astype(np.float32)
            measured_mm *= depth.get_units() * 1000.0
            smoothed_mm = smooth_depth(measured_mm, smoothed_mm)
            measured_mm = smoothed_mm
            result_mm = MAX_DEPTH_MM - measured_mm
            heatmap = colourise(measured_mm)
            x, y = selected
            patch = measured_mm[max(0, y-2):y+3, max(0, x-2):x+3]
            valid = patch[patch > 0]
            median = float(np.median(valid)) if valid.size else None
            legend = np.full((480, 250, 3), 25, np.uint8)
            label(legend, f'{MAX_DEPTH_MM:.0f} - depth (mm)', 12, 25)
            ramp = np.linspace(MIN_MM, MAX_MM, 350, dtype=np.float32)[:, None]
            legend[50:400, 15:50] = colourise(np.repeat(ramp, 35, axis=1))
            for fraction in np.linspace(0, 1, 7):
                depth_value = MIN_MM + fraction * (MAX_MM - MIN_MM)
                label(legend, f'{MAX_DEPTH_MM - depth_value:.0f} mm',
                      60, 55 + round(fraction * 349))
            label(legend, 'Black = no depth', 12, 442)
            cv2.drawMarker(heatmap, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 18, 1)
            value = f'{result_mm[y, x]:.1f} mm' if measured_mm[y, x] > 0 else 'No reading'
            footer = np.full((125, 1530, 3), 25, np.uint8)
            label(footer, f'Pixel ({x}, {y}): {value}', 12, 25)
            label(footer, 'Click depth | F: focus | W: wide | R: reset | Q/Esc: close', 12, 101)
            rgb = np.asanyarray(color.get_data())
            cv2.imshow(WINDOW, np.vstack((np.hstack((rgb, heatmap, legend)), footer)))
            key = cv2.waitKey(1) & 0xff
            if key == ord('f') and median is not None:
                MIN_MM, MAX_MM = max(1, median - 100), median + 100
            elif key == ord('w'):
                MIN_MM, MAX_MM = 200, 2000
            elif key == ord('r'):
                MIN_MM, MAX_MM = 800, 1500
            if key in (27, ord('q')):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if started:
            pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
