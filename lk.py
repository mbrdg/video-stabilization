from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import cv2


def optical_flow(video: str) -> None:
    """Computes optical-flow using Lucas-Kanade Method"""

    shi_tomasi_params = {
        "maxCorners": 100,
        "qualityLevel": 0.3,
        "minDistance": 7,
        "blockSize": 7,
    }

    lk_params = {
        "winSize": (15, 15),
        "maxLevel": 2,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    }

    rng = np.random.default_rng()
    tracking_colors = rng.integers(0, 255, size=(100, 3))

    capture = cv2.VideoCapture(video)

    ret, prev_frame = capture.read()
    prev_gray_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray_frame, mask=None, **shi_tomasi_params)

    drawing_mask = np.zeros_like(prev_frame)

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            print("info: finished reading the capture")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray_frame, gray_frame, prev_pts, None, **lk_params
        )

        if curr_pts is None:
            continue

        next_pts = curr_pts[status == 1]
        curr_pts = prev_pts[status == 1]

        # homography, status = cv2.findHomography(curr_pts, next_pts, cv2.RANSAC)
        # cv2.decomposeHomographyMat(homography, )
        # print(homography)

        for i, (new, old) in enumerate(zip(next_pts, curr_pts)):
            a, b = new.ravel()
            c, d = old.ravel()
            drawing_mask = cv2.line(
                drawing_mask,
                (int(a), int(b)),
                (int(c), int(d)),
                tracking_colors[i].tolist(),
                2,
            )
            frame = cv2.circle(
                frame, (int(a), int(b)), 5, tracking_colors[i].tolist(), -1
            )

        img = cv2.add(frame, drawing_mask)
        cv2.imshow("frame", img)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break

        prev_gray_frame = gray_frame.copy()
        prev_pts = curr_pts.reshape(-1, 1, 2)

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = ArgumentParser(prog="lk", description="Lucas-Kanade Optical Flow")
    parser.add_argument("-i", "--input", help="input video file", required=True)

    args = parser.parse_args()

    optical_flow(args.input)
