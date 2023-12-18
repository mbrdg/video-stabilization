# gauss_stab.py
import argparse
import pathlib

import numpy as np
import cv2
from scipy import ndimage, signal


SIGMA_MAT = np.array([
    [1000, 750, 700],
    [750, 1000, 700]
])
UPSCALING = 0.20


def homography_generator(transforms):
    h = np.identity(3)
    wsp = np.dstack((transforms[:, 0, :], transforms[:, 1, :], np.array([[0, 0, 1]] * transforms.shape[0])))

    for i in range(wsp.shape[0]):
        h = wsp[i].T @ h
        yield np.linalg.inv(h)


def smooth(trajectory, sigma):
    x, y = trajectory.shape[1:]

    smoothed_trajectory = np.zeros_like(trajectory)

    for i in range(x):
        for j in range(y):
            kernel = signal.windows.gaussian(10000, sigma[i, j])
            kernel /= np.sum(kernel)
            smoothed_trajectory[:, i, j] = ndimage.convolve(trajectory[:, i, j], kernel, mode="reflect")

    return smoothed_trajectory


def fix_border(frame):
    s = frame.shape

    t = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.0 + UPSCALING)
    frame = cv2.warpAffine(frame, t, (s[1], s[0]))

    return frame


def fix_border2(shape, transforms):
    w, h = shape
    
    corners = np.array([
        [0, 0, 1], 
        [h, 0, 1], 
        [0, w, 1], 
        [h, w, 1]
    ]).T

    maxmin = []

    prev_transform = np.identity(3)
    for transform in transforms:
        transform = np.concatenate((transform, np.array([[0, 0, 1]])))
        transform = transform @ prev_transform
        
        new_corners = np.linalg.inv(transform) @ corners
        xmin, xmax = np.min(new_corners[0]), np.max(new_corners[0])
        ymin, ymax = np.min(new_corners[1]), np.max(new_corners[1])
        maxmin.extend(((ymax, xmax), (ymin, xmin)))

        prev_transform = transform.copy()

    maxmin = np.array(maxmin)
    top = np.min(maxmin[:, 0])
    bot = np.max(maxmin[:, 0])
    left = np.min(maxmin[:, 1])
    right = np.max(maxmin[:, 1])
    
    return int(-top), int(bot - w), int(-left), int(right - h)


def main():
    parser = argparse.ArgumentParser(prog="gstab", description="video stabilizer")
    parser.add_argument("-i", "--input", required=True, help="video file input")
    args = parser.parse_args()

    input = pathlib.Path(args.input)
    cap = cv2.VideoCapture(str(input))

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    _, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    transforms = np.full((n_frames - 1, 2, 3), np.eye(2, 3), dtype=np.float32)

    for i in range(n_frames - 2):
        ret, curr = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        template_img = prev_gray.copy()
        input_img = curr_gray.copy()
        transforms[i] = cv2.findTransformECC(
            template_img,
            input_img,
            transforms[i],
            cv2.MOTION_EUCLIDEAN
        )[1]

        prev_gray = curr_gray

    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smooth(trajectory, SIGMA_MAT)

    conv = lambda m: ndimage.convolve(m, [0, 1, -1], mode="reflect")
    transforms_smooth = np.apply_along_axis(conv, 0, smoothed_trajectory)
    transforms_smooth = transforms.astype(np.float64) - transforms_smooth.astype(np.float64)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    homography = homography_generator(transforms)
    _, frame = cap.read()

    cv2.imshow("gstab", frame)
    cv2.waitKey(30)

    for i in range(1, n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        h_integral = next(homography)
        frame_stabilized = cv2.warpPerspective(frame, h_integral, (w, h))
        # frame_stabilized = fix_border(frame_stabilized)
        
        frame_out = cv2.hconcat([frame, frame_stabilized])

        if (frame_out.shape[1] > 1920):
            frame_out = cv2.resize(frame_out, (frame_out.shape[1] // 2, frame_out.shape[0] // 2))

        cv2.imshow("gstab", frame_out)
        cv2.waitKey(30)

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
