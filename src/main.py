# main.py
from argparse import ArgumentParser, BooleanOptionalAction

import numpy as np
import cv2

from smoothing import low_pass_filter
from plotting import plot_trajectories_from_solver, plot_transformations, plot_trajectories
from solve import stabilize
from output import output_from_filter, output_from_solver


def get_transformations_between_frames(
    capture: cv2.VideoCapture, *, solver: bool
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
        # Shi-Tomasi (or Harris) method to determine good corners to track
        old_corners = cv2.goodFeaturesToTrack(
            prev_gray_frame,
            mask=None,
            maxCorners=200,
            qualityLevel=0.1,
            minDistance=30,
            blockSize=3,
        )

        success, curr_frame = capture.read()
        if not success:
            print("info: finished reading capture")
            break

        # Lucas-Kanade method to determine the optical flow
        curr_gray_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        new_corners, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray_frame,
            curr_gray_frame,
            old_corners,
            None,
        )

        if new_corners is None:
            print(f"warn: unable to detect features in frame {frame_idx}")
            continue

        # Filtering valid points that can be tracked
        good_points = np.where(status == 1)[0]
        new_corners = new_corners[good_points]
        old_corners = old_corners[good_points]

        matrix, _ = cv2.estimateAffine2D(new_corners, old_corners)

        if solver:
            transformations[frame_idx + 1, :, :2] = matrix.T
        else:
            # affine matrix decomposition [dx, dy, da]
            transformations[frame_idx] = np.array([
                matrix[0, 2], matrix[1, 2], np.arctan2(matrix[0, 1], matrix[0, 0])
            ])

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
    smoothed_trajectory = low_pass_filter(trajectory, radius=50, order=2)

    if plot:
        plot_trajectories(trajectory, smoothed_trajectory)

    return transforms + (smoothed_trajectory - trajectory)


def main(video_file_path: str, crop_ratio: float, *, solver: bool, plot: bool) -> None:
    assert 0.0 < crop_ratio < 1.0

    capture = cv2.VideoCapture(video_file_path)
    frame_shape = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    transformations = get_transformations_between_frames(capture, solver=solver)
    if plot:
        plot_transformations(transformations, smoothed=False)

    if solver:
        inv_crop_ratio = 1.0 - crop_ratio
        status, smoothed_transformations = stabilize(
            transformations, frame_shape, inv_crop_ratio
        )
        
        if plot and status == 1:
            plot_trajectories_from_solver(transformations, smoothed_transformations)

        if status == 1:
            output_from_solver(capture, smoothed_transformations, inv_crop_ratio)
            capture.release()
            return None

        else:
            print("warn: solution did not converge, using default method")

    smoothed_transforms = motion_compensation(transformations, plot=plot)
    if plot:
        plot_transformations(smoothed_transforms, smoothed=True)

    output_from_filter(capture, smoothed_transforms, crop_ratio)
    capture.release()


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="vstab", description="Video Stabilization", epilog="computer vision @ VUT"
    )
    parser.add_argument("-i", "--input", help="input video file", required=True)
    parser.add_argument("-c", "--crop-ratio", type=float, required=True)
    parser.add_argument("--solver", action=BooleanOptionalAction)
    parser.add_argument("--plot", action=BooleanOptionalAction)

    args = parser.parse_args()

    main(args.input, args.crop_ratio, solver=args.solver, plot=args.plot)
