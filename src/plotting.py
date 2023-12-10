# plotting.py
import numpy as np
import matplotlib.pyplot as plt


def plot_transformations(transformations: np.ndarray, *, smoothed: bool) -> None:
    """
    Utilitary function to plot the transformation components for each pair of
    consecitive frames.

    Params:
    -------
    transformations -- array containing the values of each transformation.
    smoothed -- True if passed transformations are smoothed otherwise should be
                False, only used for plot title displaying purposes.
    """
    x = np.arange(transformations.shape[0])

    fig, ax = plt.subplots()
    title = "Smoothed Transformations" if smoothed else "Original Transformations"
    fig.suptitle(title)

    ax.set_xlabel("frame number")
    ax.set_ylabel("delta pixels")

    ax.plot(x, transformations[:, 0], label="dx")
    ax.plot(x, transformations[:, 1], label="dy")
    ax.plot(x, transformations[:, 2], label="da")

    ax.legend(loc="upper right")
    plt.show()


def plot_trajectories(trajectory: np.ndarray, smoothed_trajectory: np.ndarray) -> None:
    """
    Utilitary function to plot the trajectory components along the video.

    Params:
    -------
    trajectory -- array containing the values of the original trajectory of the
                  relevant components (x, y, angle)
    smoothed_trajectory --  array containing the values of the smoothed
                            trajectory of the relevant components (x, y, angle)
    """
    assert trajectory.shape == smoothed_trajectory.shape

    x = np.arange(trajectory.shape[0])

    fig, (ax_dx, ax_dy, ax_da) = plt.subplots(nrows=3, ncols=1)
    fig.suptitle("Trajectory")

    def axis_plot(ax, component, smoothed_component, label_name):
        ax.set_xlabel("frame number")
        ax.set_ylabel("delta pixels")
        ax.plot(x, component, label=label_name)
        ax.plot(x, smoothed_component, label=f"smoothed {label_name}")
        ax.legend(loc="upper right")

    axis_plot(ax_dx, trajectory[:, 0], smoothed_trajectory[:, 0], "x")
    axis_plot(ax_dy, trajectory[:, 1], smoothed_trajectory[:, 1], "y")
    axis_plot(ax_da, trajectory[:, 2], smoothed_trajectory[:, 2], "a")

    plt.show()


def plot_trajectories_from_solver(trajectory: np.ndarray, smoothed_trajectory: np.ndarray) -> None:
    """
    Utilitary function to plot the trajectory components along the video when
    smoothing is done using the solver. Otherwise, use `plot_trajectories`.

    Params:
    -------
    trajectory -- array containing the values of the original trajectory of the
                  relevant components
    smoothed_trajectory --  array containing the values of the smoothed
                            trajectory of the relevant components
    """
    assert trajectory.shape == smoothed_trajectory.shape
    number_of_frames = trajectory.shape[0]

    unstable_trajectory = trajectory.copy()
    for frame in range(1, number_of_frames):
        unstable_trajectory[frame, :, :] = unstable_trajectory[frame - 1, :, :] @ trajectory[frame, :, :]
    
    stable_trajecotry = unstable_trajectory.copy()
    for frame in range(number_of_frames):
        stable_trajecotry[frame, :, :] = unstable_trajectory[frame, :, :] @ smoothed_trajectory[frame, :, :]

    origin = np.array([0, 0, 1])
    plot_trajectories(origin @ unstable_trajectory, origin @ stable_trajecotry)

