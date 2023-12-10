# smoothing.py
import numpy as np
from numpy.fft import fft2, ifft2
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


def wiener_filter_1(frame):
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


def wiener_filter_2(frame):  # somos capazes de ter de meter a imagem a cinzento
    psf = np.ones((5, 5)) / 25
    img = convolve2d(frame, psf, "same")
    img += 0.1 * img.std() * np.random.standard_normal(img.shape)
    return restoration.wiener(img, psf, 1100)
