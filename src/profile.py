# profile.py
from glob import iglob
from pathlib import Path
import os
from types import SimpleNamespace
import time

import numpy as np
import matplotlib.pyplot as plt

import stab


def plot(frames, elapsed):
    fig, ax = plt.subplots(dpi=600, layout="tight")
    fig.suptitle(f"Linear Regression over time taken to stabilize")

    ax.scatter(frames, elapsed)

    series = np.polynomial.polynomial.Polynomial.fit(frames, elapsed, 1)
    x, y = series.linspace()
    ax.plot(x, y, label=f"{series:unicode}", color='r', linewidth=2)

    ax.legend()

    ax.set_xlabel("lenght [frames]")
    ax.set_ylabel("time [s]")

    plots = Path("plots/")
    plots.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots / "time.pdf")
    

def main():
    videos_glob = "data/unstable/*.avi"
    videos = [v for v in iglob(videos_glob) if os.path.isfile(v)]

    frames = np.empty(len(videos))
    elapsed = np.empty(len(videos))

    args = SimpleNamespace()
    args.plot = False
    args.output = ""

    for i, video in enumerate(videos):
        args.input = video
        
        start = time.perf_counter()
        n_frames = stab.main(args)
        end = time.perf_counter()
        
        frames[i] = n_frames
        elapsed[i] = end - start

    plot(frames, elapsed)


if __name__ == "__main__":
    main()
