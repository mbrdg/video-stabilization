# stab.py
import argparse
import pathlib

import cv2
import numpy as np
import matplotlib.pyplot as plt


SMOOTHING_RADIUS = 50
UPSCALING = 0.20


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


def fix_border(frame):
    s = frame.shape

    t = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.0 + UPSCALING)
    frame = cv2.warpAffine(frame, t, (s[1], s[0]))

    return frame


def plot_trajectories(original, smoothed, file):
    assert original.shape == smoothed.shape

    frames = np.arange(original.shape[0])

    fig, ax = plt.subplots(nrows=3, ncols=1, dpi=600, layout="tight")
    fig.suptitle(f"Trajectories over video no. {file.stem}")

    ax[0].plot(frames, original[:, 0], label="original")
    ax[0].plot(frames, smoothed[:, 0], label="smooth")
    ax[0].set_xlabel("frames")
    ax[0].set_ylabel("$x$ [px]")
    ax[0].legend(loc="upper right")

    ax[1].plot(frames, original[:, 1], label="original")
    ax[1].plot(frames, smoothed[:, 1], label="smooth")
    ax[1].set_xlabel("frames")
    ax[1].set_ylabel("$y$ [px]")
    ax[1].legend(loc="upper right")

    ax[2].plot(frames, original[:, 2], label="original")
    ax[2].plot(frames, smoothed[:, 2], label="smooth")
    ax[2].set_xlabel("frames")
    ax[2].set_ylabel("$\\theta$ [rad]")
    ax[2].legend(loc="upper right")

    plots = pathlib.Path("plots/")
    plots.mkdir(parents=True, exist_ok=True)
    fig.savefig((plots / file.stem).with_suffix(".pdf"))


def main(args):
    vid = pathlib.Path(args.input)
    cap = cv2.VideoCapture(str(vid))

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
            print("warn: unable to generate transformation")
            transforms[i] = np.zeros(3)
        else:
            dx = m[0, 2]
            dy = m[1, 2]
            da = np.arctan2(m[1, 0], m[0, 0])
            transforms[i] = [dx, dy, da]

        prev_gray = curr_gray

    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smooth(trajectory)

    if args.plot:
        plot_trajectories(trajectory, smoothed_trajectory, vid)

    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    writer = None
    if args.output:
        out = pathlib.Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out), fourcc, fps, (w, h // 2))

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

        if writer is not None:
            writer.write(frame_out)

        cv2.imshow("stab", frame_out)
        cv2.waitKey(30)

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()
    cap.release()

    return n_frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="stab", description="video stabilizer")
    parser.add_argument("-i", "--input", required=True, help="video file input")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, help="plot trajectories")
    parser.add_argument("-o", "--output", help="video file output")
    args = parser.parse_args()

    main(args)
