import numpy as np
import cv2
from numpy.fft import fft2, ifft2
from skimage import restoration
from scipy.signal import convolve2d


def moving_average_filter(
        trajectory: np.ndarray, iterations: int = 3, radius: int = 30
    ) -> np.ndarray:
    """
    Moving average filter.
    Steps:
        1. Create the filter
        2. For iteration in iterations
            1. Pad the curve in the edges by the radius
            2. Convolution with the filter
            3. Remove the padding
    
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
    f = np.full(window_size, 1 / window_size)

    def apply(curve):
        padded = np.lib.pad(curve, (radius, radius), "edge")
        smoothed = np.convolve(padded, f, mode="same")
        return smoothed[radius:-radius]

    for _ in range(iterations):
        smoothed_trajectory[:, 0] = apply(smoothed_trajectory[:, 0])
        smoothed_trajectory[:, 1] = apply(smoothed_trajectory[:, 1])
        smoothed_trajectory[:, 2] = apply(smoothed_trajectory[:, 2])

    return smoothed_trajectory


def low_pass_filtering_2(frame):
    # prepare the 5x5 shaped filter
    kernel = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )
    kernel = kernel / sum(kernel)

    return cv2.filter2D(frame, -1, kernel)


def low_pass_filtering_3(frame):
    return cv2.GaussianBlur(frame, (5, 5), 0)


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


