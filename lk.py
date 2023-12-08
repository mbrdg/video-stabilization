from pathlib import Path

import numpy as np
import cv2


def optical_flow(video: Path) -> None:
    """Computes optical-flow using Lucas-Kanade Method"""

    shi_tomasi_params = {
        'maxCorners': 100,
        'qualityLevel': 0.3,
        'minDistance': 7,
        'blockSize': 7
    }

    lk_params = {
        'win_size': (15, 15),
        'maxLevel': 2,
        'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    }

    rng = np.random.default_rng()
    tracking_colors = rng.integers(0, 255, size=(100, 3))

    capture = cv2.VideoCapture(str(video))

    ret, prev_frame = capture.read()
    prev_gray_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(prev_gray_frame, mask=None, **shi_tomasi_params)

    drawing_mask = np.zeros_like(prev_frame)

    while True:
        ret, frame = capture.read()
        if not ret:
            print('Finished reading the capture')
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray_frame, gray_frame, corners, None, **lk_params)

        if flow is None:
            continue

        new_corners = flow[st == 1]
        old_corners = corners[st == 1]

        for i, (new, old) in enumerate(zip(new_corners, old_corners)):
            a, b = new.ravel()
            c, d = old.ravel()
            drawing_mask = cv2.line(
                drawing_mask, (int(a), int(b)), (int(c), int(d)), 
                tracking_colors[i].toList(), -1
            )
            frame = cv2.circle(frame, drawing_mask)

        img = cv2.add(frame, drawing_mask)
        cv2.imshow('frame', img)

        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break

        prev_gray_frame = gray_frame.copy()
        corners = new_corners.reshape(-1, 1, 2)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    data_dir = Path('data')
    optical_flow(data_dir / 'unsatble' / '42.avi')
