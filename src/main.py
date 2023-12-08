# main.py
# Authors: xfranc00, xboave00
from argparse import ArgumentParser
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torchvision.transforms.functional as F
from torchvision.io import read_video
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from torchvision.utils import flow_to_image


# globals
plt.rcParams["savefig.bbox"] = "tight"
weights = Raft_Small_Weights.DEFAULT
transforms = weights.transforms()


def parse_arguments():
    """Parses the command line arguments"""
    parser = ArgumentParser(
        description="Optical flow computation using the RAFT model from PyTorch"
    )

    parser.add_argument("--file", required=True, type=str, help="input video file")

    parser.add_argument(
        "--frames", type=int, default=-1, help="number of frames to be processed"
    )

    parser.add_argument(
        "-W",
        "--width",
        type=int,
        default=520,
        help="width of the output frames (must be divisible by 8)",
    )
    parser.add_argument(
        "-H",
        "--height",
        type=int,
        default=960,
        help="height of the output frames (must be divisible by 8)",
    )

    return parser.parse_args()


def preprocess(first_batch, follow_batch, size):
    # TODO: docstring
    first_batch = F.resize(first_batch, size=size, antialias=False)
    follow_batch = F.resize(follow_batch, size=size, antialias=False)
    return transforms(first_batch, follow_batch)


def main():
    args = parse_arguments()

    # Read and preprocess the video batches (pairs of consecutive frames)
    frames, _, _ = read_video(args.file, output_format="TCHW", pts_unit="sec")
    first_batch, follow_batch = preprocess(
        frames[0 : args.frames], frames[1 : args.frames + 1], (args.width, args.height)
    )
    print(f"shape = {first_batch.shape}, dtype = {first_batch.dtype}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Instatiate the pretrained model and predict the flow
    model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(device)
    model = model.eval()

    list_of_flows = model(first_batch.to(device), follow_batch.to(device))
    print(f"type = {type(list_of_flows)}")
    print(f"lenght = {len(list_of_flows)} (number of iterations of the model)")

    # Get the flow from the last iteration (the most accurate)
    predicted_flows = list_of_flows[-1]
    print(f"dtype = {predicted_flows.dtype}")
    print(f"shape = {predicted_flows.shape} (N, 2, H, W)")
    print(f"min = {predicted_flows.min()}, max = {predicted_flows.max()}")

    # Animation with the flow based on a input video
    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    ax[1].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    flow_imgs = flow_to_image(predicted_flows)
    first_batch = [(img + 1) / 2 for img in first_batch]

    def animate_callback(i):
        original_img = F.to_pil_image(first_batch[i].to("cpu"))
        flow_img = F.to_pil_image(flow_imgs[i].to("cpu"))
        ax[0].imshow(np.asarray(original_img))
        ax[1].imshow(np.asarray(flow_img))
        return ax

    animation.FuncAnimation(
        fig, animate_callback, repeat=False, frames=flow_imgs.shape[0]
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
