# stab.py
import argparse

import numpy as np
import cv2


SMOOTHING_RADIUS = 50


def moving_average(curve, radius):
    window_size = 2 * radius + 1
    f = np.full(window_size, 1 / window_size)

    curve_pad = np.lib.pad(curve, (radius, radius), "edge")
    curve_smoothed = np.convolve(curve_pad, f, mode="same")
    curve_smoothed = curve_smoothed[radius:-radius]

    return curve_smoothed


def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)

    smoothed_trajectory[:, 0] = moving_average(trajectory[:, 0], radius=SMOOTHING_RADIUS)
    smoothed_trajectory[:, 1] = moving_average(trajectory[:, 1], radius=SMOOTHING_RADIUS)
    smoothed_trajectory[:, 2] = moving_average(trajectory[:, 2], radius=SMOOTHING_RADIUS)

    return smoothed_trajectory


def fix_border(frame, upscale=0.20):
    s = frame.shape

    t = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1 + upscale)
    frame = cv2.warpAffine(frame, t, (s[1], s[0]))

    return frame


def main():
    parser = argparse.ArgumentParser(prog="stab", description="video stabilizer")
    parser.add_argument("-i", "--input", required=True, help="input video file")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    _, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    transforms = np.zeros((n_frames - 1, 3))

    for i in range(n_frames - 2):
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=200,
            qualityLevel=0.05,
            minDistance=30,
            blockSize=3,
        )

        ret, curr = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        assert prev_pts.shape == curr_pts.shape
        
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]

        if m is None:
            print("info: unable to generate transformation because frame was to dark")
            transforms[i] = np.zeros(3)
        else:
            dx = m[0, 2]
            dy = m[1, 2]
            da = np.arctan2(m[1, 0], m[0, 0])
            transforms[i] = [dx, dy, da]

        prev_gray = curr_gray

    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smooth(trajectory)

    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for i in range(n_frames - 2):
        ret, frame = cap.read()
        if not ret:
            break

        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        da = transforms_smooth[i, 2]

        m = np.array([
            [np.cos(da), -np.sin(da), dx],
            [np.sin(da), np.cos(da),  dy]
        ])

        frame_stabilized = cv2.warpAffine(frame, m, (w, h))
        frame_stabilized = fix_border(frame_stabilized)

        frame_out = cv2.hconcat([frame, frame_stabilized])

        if (frame_out.shape[1] > 1920):
            frame_out = cv2.resize(frame_out, (frame_out.shape[1] // 2, frame_out.shape[0] // 2))

        cv2.imshow("stab", frame_out)
        cv2.waitKey(30)

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
