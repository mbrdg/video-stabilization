# output.py
import numpy as np
import cv2

from smoothing import wiener_filter


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


def output_from_filter(
    capture: cv2.VideoCapture, smoothed_transforms: np.ndarray, crop_ratio: float,
    *, deblur: bool = False
) -> None:
    """
    Outputs the original frame and stabilized frames side-by-side.

    Params:
    -------
    capture -- video capture object that contains the frame stream
    smoothed_transforms -- array of transforms after being smoothed
    crop_ratio -- fraction of the frame to be cropped in the output frame
    """
    number_of_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for frame_idx in range(number_of_frames - 2):
        success, frame = capture.read()
        if not success:
            print("warn: unable to grab frame from video capture")

        dx = smoothed_transforms[frame_idx, 0]
        dy = smoothed_transforms[frame_idx, 1]
        da = smoothed_transforms[frame_idx, 2]
        transformation = np.array(
            [
                [np.cos(da), -np.sin(da), dx],
                [np.sin(da), np.cos(da), dy],
            ]
        )

        w, h, _ = frame.shape
        stabilized_frame = cv2.warpAffine(frame, transformation, (h, w))
        stabilized_frame = fix_border(stabilized_frame, crop_ratio)

        img = np.hstack(
            (
                cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
                cv2.resize(
                    stabilized_frame,
                    (0, 0),
                    fx=0.5,
                    fy=0.5,
                    interpolation=cv2.INTER_AREA,
                ),
            )
        )
        
        if deblur:
            wiener_filter(frame)

        cv2.imshow("stabilized video", img)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break

    cv2.destroyWindow("stabilized video")


def output_from_solver(
    capture: cv2.VideoCapture, smoothed_transforms: np.ndarray, inv_crop_ratio: float,
    *, deblur: bool = False
) -> None:
    """
    Outputs the original frame and stabilized frame side-by-side.

    Params:
    -------
    capture -- video capture object that contains the frame stream
    smoothed_transforms -- array of transforms after being smoothed
    inv_crop_ratio -- inverse crop ratio, i.e., `1 - desired_crop_ratio`, see output()
    """

    number_of_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for frame_idx in range(number_of_frames - 2):
        success, frame = capture.read()
        if not success:
            print("warn: unable to grab frame from video capture")

        scaling_matrix = np.array(
            [
                [1.0 / inv_crop_ratio, 0.0, 0.0],
                [0.0, 1.0 / inv_crop_ratio, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        w, h, _ = frame.shape

        shifting_to_center_matrix = np.array(
            [
                [1.0, 0.0, -w / 2.0],
                [0.0, 1.0, -h / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )

        shifting_back_matrix = np.array(
            [
                [1.0, 0.0, w / 2.0],
                [0.0, 1.0, h / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )

        transform_matrix = np.eye(3)
        transform_matrix[:2][:] = smoothed_transforms[frame_idx, :, :2].T
        transformation = (
            shifting_back_matrix
            @ scaling_matrix
            @ shifting_to_center_matrix
            @ np.linalg.inv(transform_matrix)
        )

        stabilized_frame = cv2.warpAffine(frame, transformation[:2, :], (h, w))

        img = np.hstack(
            (
                cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
                cv2.resize(
                    stabilized_frame,
                    (0, 0),
                    fx=0.5,
                    fy=0.5,
                    interpolation=cv2.INTER_AREA,
                ),
            )
        )
    
        if deblur:
            wiener_filter(frame)

        cv2.imshow("stabilized video", img)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break

    cv2.destroyWindow("stabilized video")
