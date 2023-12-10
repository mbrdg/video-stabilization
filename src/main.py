# main.py
from argparse import ArgumentParser, BooleanOptionalAction

import numpy as np
import cv2

from smoothing import low_pass_filter
from plotting import plot_transformations, plot_trajectories
from solve import stabilize


def get_transformations_between_frames(
        capture: cv2.VideoCapture,
        *, solver: bool
    ) -> np.ndarray:
    """
    Gets the transformations between frames using sparse Lucas-Kanade optical 
    flow method. Resets the capture back to the initial frame before finishing.

    Params:
    -------
    capture: a VideoCapture object with the video to be stabilized.

    Returns:
    --------
    An array containing the decomposed affine transformations between frames.
    """
    number_of_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    transformations = np.empty(shape=(number_of_frames - 1, 3))
    if solver:
        transformations = np.full((number_of_frames, 3, 3), np.eye(3))
    
    success, prev_frame = capture.read()
    prev_gray_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    for frame_idx in range(number_of_frames - 2):
        # Shi-Tomasi method to determine goos corners to track
        old_corners = cv2.goodFeaturesToTrack(
            prev_gray_frame, mask=None,
            maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7
        )

        success, curr_frame = capture.read()
        if not success:
            print("info: finished reading capture")
            break

        # Lucas-Kanade method to determine the optical flow
        curr_gray_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        new_corners, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray_frame, curr_gray_frame, old_corners, None,
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        if new_corners is None:
            print(f"warn: unable to detect features in frame {frame_idx}")
            continue

        # Filtering valid points that can be tracked
        good_points = np.where(status == 1)[0]
        new_corners = new_corners[good_points]
        old_corners = old_corners[good_points]

        if solver:
            matrix, _ = cv2.estimateAffine2D(new_corners, old_corners)

            if matrix is not None:
                transformations[frame_idx + 1, :, :2] = matrix.T

            prev_gray_frame = curr_gray_frame
            continue
        
        matrix = cv2.estimateAffine2D(new_corners, old_corners)

        # affine matrix decomposition
        # dx, dy represent the translation components and da the rotation component
        if matrix is None:
            dx = matrix[0, 0, 2]
            dy = matrix[0, 1, 2]
            da = np.arctan2(matrix[0, 0, 1], matrix[0, 0, 0])
        else:
            dx, dy, da = 0, 0, 0

        transformations[frame_idx] = [dx, dy, da]

        prev_gray_frame = curr_gray_frame
    
    return transformations


def motion_compensation(transforms: np.ndarray, *, plot: bool) -> np.ndarray:
    """
    Performs the motion compensation procedure by applying a low pass filter
    on top of the transformations between frames.

    Params:
    -------
    transforms -- Affine transformation matrix between each pair of consecutive frames.
    plot -- Plots the trajectory (cumulative sum of transformations) if True

    Returns:
    --------
    An array with the same shape as transforms but smoothed.
    """
    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = low_pass_filter(trajectory)

    if plot:
        plot_trajectories(trajectory, smoothed_trajectory)

    return transforms + (smoothed_trajectory - trajectory)


def fix_border(frame: np.ndarray, crop_ratio: float) -> np.ndarray:
    """
    Applies upscaling in order to hide black borders after applying the warpAffine
    with the smoothen frame transformation. It create a rotation matrix focused
    on the center of the frame and applies some scaling w.r.t. crop ratio.

    Params:
    -------
    frame -- frame from the original video to be upscalled after applied transformation.
    crop_ratio -- value in [0.0, 1.0] that represents the upscaling factor.

    Returns:
    --------
    Upscaled frame that should be part of the output video
    """
    w, h, _ = frame.shape
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1.0 + crop_ratio)
    frame = cv2.warpAffine(frame, rotation_matrix, (h, w))
    return frame


def output(
        capture: cv2.VideoCapture,
        smoothed_transforms: np.ndarray,
        crop_ratio: float
    ) -> None:
    number_of_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    for frame_idx in range(number_of_frames - 2):
        success, frame = capture.read()
        if not success:
            print("warn: unable to grab frame from video capture")

        dx = smoothed_transforms[frame_idx, 0]
        dy = smoothed_transforms[frame_idx, 1]
        da = smoothed_transforms[frame_idx, 2]
        transformation = np.array([
            [np.cos(da), -np.sin(da), dx],
            [np.sin(da), np.cos(da), dy],
        ])

        w, h, _ = frame.shape
        stabilized_frame = cv2.warpAffine(frame, transformation, (h, w))
        stabilized_frame = fix_border(stabilized_frame, crop_ratio)

        img = np.hstack((
            cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
            cv2.resize(stabilized_frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        ))

        # TODO: wiener filter to remove deblurring

        cv2.imshow("stabilized video", img)
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break

    cv2.destroyWindow("stabilized video")


def main(
        video_file_path: str,
        crop_ratio: float,
        *, solver: bool, plot: bool
    ) -> None:
    assert 0.0 < crop_ratio < 1.0

    capture = cv2.VideoCapture(video_file_path)
    frame_shape = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )

    transformations = get_transformations_between_frames(capture, solver=solver)
    if plot:
        plot_transformations(transformations, smoothed=False)

    if solver:
        status, smoothed_transforms = stabilize(transformations, frame_shape, 1.0 - crop_ratio)
        if status != 1:
            print("warn: solution did not converge, fallback to default method")
            smoothed_transforms = motion_compensation(transformations, plot=False)
    else:
        smoothed_transforms = motion_compensation(transformations, plot=plot)

    if plot:
        plot_transformations(smoothed_transforms, smoothed=True)

    output(capture, smoothed_transforms, crop_ratio)

    capture.release()


if __name__ == "__main__":
    parser = ArgumentParser(prog="vstab",
                            description="Video Stabilization", 
                            epilog="computer vision @ VUT")
    parser.add_argument("-i", "--input", help="input video file", required=True)
    parser.add_argument("-c", "--crop-ratio", type=float, required=True)
    parser.add_argument("--solver", action=BooleanOptionalAction)

    args = parser.parse_args()

    main(args.input, args.crop_ratio, solver=args.solver, plot=False)
