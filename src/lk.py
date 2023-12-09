from argparse import ArgumentParser

import numpy as np
import cv2
from numpy.fft import fft2, ifft2

import smoothing


def main(video: str) -> None:
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

    capture = cv2.VideoCapture(video)
    number_of_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    transformations = np.empty((number_of_frames - 1, 3))

    ret, prev_frame = capture.read()
    prev_gray_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray_frame, mask=None, **shi_tomasi_params)

    while capture.isOpened():
        ret, frame = capture.read()
        current_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))

        if not ret or number_of_frames - current_frame < 2:
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

        affine = cv2.estimateAffine2D(curr_pts, next_pts)
        dx = affine[0][0][2]
        dy = affine[0][1][2]
        da = np.arctan2(affine[0][1][0], affine[0][0][0])

        current_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        transformations[current_frame] = [dx, dy, da]

        prev_gray_frame = gray_frame.copy()
        prev_pts = curr_pts.reshape(-1, 1, 2)

        # cv2.imshow("frame", frame)
        # key = cv2.waitKey(30) & 0xFF
        # if key == 27:
        #    break

    smoothed_transforms = motion_compensation(transformations)

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while capture.isOpened():
        ret, frame = capture.read()
        current_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))

        if current_frame == 0:
            continue

        if not ret or number_of_frames - current_frame < 2:
            print("info: finished reading the capture")
            break

        dx = smoothed_transforms[current_frame, 0]
        dy = smoothed_transforms[current_frame, 1]
        da = smoothed_transforms[current_frame, 2]
        transformation_matrix = np.array(
            [
                [np.cos(da), -np.sin(da), dx],
                [np.sin(da), np.cos(da), dy],
            ]
        )

        w, h, _ = frame.shape
        stabilized_frame = cv2.warpAffine(frame, transformation_matrix, (w, h))
        img = crop(stabilized_frame)

        cv2.imshow("frame", img)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


def motion_compensation(transforms: np.ndarray) -> np.ndarray:
    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)

    # Smooth trajectory using moving average filter
    smoothed_trajectory = smoothing.smooth(trajectory, smoothing.SMOOTHING_RADIUS)

    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory

    return transforms + difference


def crop(frame: np.ndarray, crop_ratio: float = 0.04) -> np.ndarray:
    w, h, _ = frame.shape
    rotation_matrix = cv2.getRotationMatrix2D((h // 2, w // 2), 0, 1.0 + crop_ratio)
    return cv2.warpAffine(frame, rotation_matrix, (h, w))


def low_pass_filtering_1(frame):
    # Initialize variables
    homoFiltered = np.eye(3, dtype=np.float32)
    alpha = 0.9
    a1 = np.eye(3, dtype=np.float32) * alpha
    a2 = np.eye(3, dtype=np.float32) * (1.0 - alpha)

    # Assuming you have a function CalcHomography that returns the homography matrix
    homo = cv2.findHomography(frame)  # Is wrong falta parametros

    # Update filtered homography
    homoFiltered = np.dot(a1, np.dot(homoFiltered, homo)) + np.dot(a2, homo)

    # Apply stabilized transformation to the frame
    stabilized_frame = cv2.warpPerspective(
        frame, homoFiltered, (frame.shape[1], frame.shape[0])
    )

    # Display the stabilized frame (you can also save it or perform further processing)
    cv2.imshow("Stabilized Frame", stabilized_frame)

    return stabilized_frame


def low_pass_filtering_2(frame):
    # prepare the 5x5 shaped filter
    kernel = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )
    kernel = kernel / sum(kernel)

    return cv2.filter2D(frame, -1, kernel)


def low_pass_filtering_3(frame):
    return cv2.GaussianBlur(frame, (5, 5), 0)


def wiener_filter(frame):
    kernel = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )
    K = 10

    kernel /= np.sum(kernel)
    dummy = np.copy(frame)
    dummy = fft2(dummy)
    kernel = fft2(kernel, s=frame.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.abs(ifft2(dummy))
    return dummy


if __name__ == "__main__":
    parser = ArgumentParser(prog="lk", description="Lucas-Kanade Optical Flow")
    parser.add_argument("-i", "--input", help="input video file", required=True)

    args = parser.parse_args()

    main(args.input)
