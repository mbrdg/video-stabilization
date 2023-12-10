# smoothing.py
import numpy as np
from skimage import restoration
from scipy.signal import savgol_filter
from scipy.signal import convolve2d


def low_pass_filter(
    trajectory: np.ndarray,
    iterations: int = 1,
    radius: int = 30,
    order: int = 1,
) -> np.ndarray:
    """
    Low-pass filter employing the Savitzky-Golay method.
    If `order` is set to 1 then this acts as a moving average filter.
    This function acts as a wrapper over scipy.signal.savgol_filter, thus,
    for more infor consult its own documentation.

    Params:
    -------
    trajectory -- curve to be smoothed.
    iterations -- number of iterations to execute the smoothing procedure.
    radius -- number of frames around to account for in the moving average.

    Returns:
    --------
    smoothed_trajectory -- copy of the trajectory, smoothed.
    """
    smoothed_trajectory = np.copy(trajectory)

    window_size = 2 * radius + 1
    for _ in range(iterations):
        smoothed_trajectory = savgol_filter(
            smoothed_trajectory, window_size, order, axis=0, mode="nearest"
        )

    return smoothed_trajectory


def wiener_filter(frame: np.ndarray) -> np.ndarray:
    """
    Wiener-Hunt deconvolution to deblur a frame.
    It is a wrapper 

    Params:
    -------
    frame -- frame to be deblured

    Returns:
    --------
    Deblured input frame.
    """

    psf = np.ones((5, 5)) / 25
    return restoration.wiener(frame, psf, 0.1)

