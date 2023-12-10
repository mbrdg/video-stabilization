# solve.py
from typing import Tuple

import numpy as np
import pulp


def compute_crop_window(
        frame_shape: Tuple[float, float],
        crop_ratio: float
    ) -> np.ndarray:
    """
    Computes the corners of a frame given a crop ratio.

    Params:
    -------
    frame_shape -- width and height of a frame
    crop_ratio -- the ratio of the size between the original and cropped frames

    Returns:
    -------
    An array containing the corners' position for the cropped frame
    """
    w, h = frame_shape
    center_x, center_y = round(h / 2), round(w / 2)
    crop_w, crop_h = round(h * crop_ratio), round(w * crop_ratio)
    crop_x, crop_y = round(center_x - crop_w / 2), round(center_y - crop_h / 2)
    
    return np.array([
        [crop_x,            crop_y],
        [crop_x + crop_w,   crop_y],
        [crop_x,            crop_y + crop_h],
        [crop_x + crop_w,   crop_y + crop_h],
    ])


def stabilize(
        frame_transforms: np.ndarray,
        frame_shape: Tuple[float, float],
        inv_crop_ratio: float = 0.8
    ) -> Tuple[int, np.ndarray]:

    # Predefined weights
    w1, w2, w3 = 10, 1, 100
    c = (1, 1, 100, 100, 100, 100)

    number_of_frames = frame_transforms.shape[0]
    dof = 6

    corners = compute_crop_window(frame_shape, inv_crop_ratio)
    
    problem = pulp.LpProblem("stabilize", pulp.LpMinimize)
    
    e1 = pulp.LpVariable.dicts(
        "e1", ((i, j) for i in range(number_of_frames) for j in range(dof)), lowBound=0.0
    )
    e2 = pulp.LpVariable.dicts(
        "e2", ((i, j) for i in range(number_of_frames) for j in range(dof)), lowBound=0.0
    )
    e3 = pulp.LpVariable.dicts(
        "e3",  ((i, j) for i in range(number_of_frames) for j in range(dof)), lowBound=0.0
    )

    p = pulp.LpVariable.dicts(
        "p", ((i, j) for i in range(number_of_frames) for j in range(dof))
    )

    problem += w1 * pulp.lpSum([e1[i, j] * c[j] for i in range(number_of_frames) for j in range(dof)]) + \
               w2 * pulp.lpSum([e2[i, j] * c[j] for i in range(number_of_frames) for j in range(dof)]) + \
               w3 * pulp.lpSum([e3[i, j] * c[j] for i in range(number_of_frames) for j in range(dof)])

    for ts in range(number_of_frames - 3):
        residual = [
            (p[ts + 1, 0] + frame_transforms[ts + 1, 2, 0] * p[ts + 1, 2] + frame_transforms[ts + 1, 2, 1] * p[ts + 1, 3]) - p[ts, 0], 
            (p[ts + 1, 0] + frame_transforms[ts + 1, 2, 0] * p[ts + 1, 4] + frame_transforms[ts + 1, 2, 1] * p[ts + 1, 5]) - p[ts, 1],
            (frame_transforms[ts + 1, 0, 0] * p[ts + 1, 2] + frame_transforms[ts + 1, 0, 0] * p[ts + 1, 3]) - p[ts, 2],
            (frame_transforms[ts + 1, 1, 0] * p[ts + 1, 2] + frame_transforms[ts + 1, 1, 0] * p[ts + 1, 3]) - p[ts, 3],
            (frame_transforms[ts + 1, 0, 0] * p[ts + 1, 2] + frame_transforms[ts + 1, 0, 0] * p[ts + 1, 5]) - p[ts, 4],
            (frame_transforms[ts + 1, 1, 0] * p[ts + 1, 2] + frame_transforms[ts + 1, 1, 0] * p[ts + 1, 5]) - p[ts, 5],
        ]

        residual_d1 = [
            (p[ts + 2, 0] + frame_transforms[ts + 2, 2, 0] * p[ts + 2, 2] + frame_transforms[ts + 2, 2, 1] * p[ts + 2, 3]) - p[ts + 1, 0], 
            (p[ts + 2, 0] + frame_transforms[ts + 2, 2, 0] * p[ts + 2, 4] + frame_transforms[ts + 2, 2, 1] * p[ts + 2, 5]) - p[ts + 1, 1],
            (frame_transforms[ts + 2, 0, 0] * p[ts + 2, 2] + frame_transforms[ts + 2, 0, 0] * p[ts + 2, 3]) - p[ts + 1, 2],
            (frame_transforms[ts + 2, 1, 0] * p[ts + 2, 2] + frame_transforms[ts + 2, 1, 0] * p[ts + 2, 3]) - p[ts + 1, 3],
            (frame_transforms[ts + 2, 0, 0] * p[ts + 2, 2] + frame_transforms[ts + 2, 0, 0] * p[ts + 2, 5]) - p[ts + 1, 4],
            (frame_transforms[ts + 2, 1, 0] * p[ts + 2, 2] + frame_transforms[ts + 2, 1, 0] * p[ts + 2, 5]) - p[ts + 1, 5],
        ]

        residual_d2 = [
            (p[ts + 3, 0] + frame_transforms[ts + 3, 2, 0] * p[ts + 3, 2] + frame_transforms[ts + 3, 2, 1] * p[ts + 3, 3]) - p[ts + 2, 0], 
            (p[ts + 3, 0] + frame_transforms[ts + 3, 2, 0] * p[ts + 3, 4] + frame_transforms[ts + 3, 2, 1] * p[ts + 3, 5]) - p[ts + 2, 1],
            (frame_transforms[ts + 3, 0, 0] * p[ts + 3, 2] + frame_transforms[ts + 3, 0, 0] * p[ts + 3, 3]) - p[ts + 2, 2],
            (frame_transforms[ts + 3, 1, 0] * p[ts + 3, 2] + frame_transforms[ts + 3, 1, 0] * p[ts + 3, 3]) - p[ts + 2, 3],
            (frame_transforms[ts + 3, 0, 0] * p[ts + 3, 2] + frame_transforms[ts + 3, 0, 0] * p[ts + 3, 5]) - p[ts + 2, 4],
            (frame_transforms[ts + 3, 1, 0] * p[ts + 3, 2] + frame_transforms[ts + 3, 1, 0] * p[ts + 3, 5]) - p[ts + 2, 5],
        ]

        # smoothness constraints
        for d in range(dof):
            problem += -e1[ts, d] <= residual[d] <= e1[ts, d]
            problem += -e2[ts, d] <= residual_d1[d] - residual[d] <= e2[ts, d]
            problem += -e3[ts, d] <= residual_d2[d] - 2 * residual_d1[d] + residual[d] <= e3[ts, d]

    w, h = frame_shape
    for ts in range(number_of_frames):
        # proximity constraints
        problem += 0.9 <= p[ts, 2] <= 1.0
        problem += -0.1 <= p[ts, 3] <= 0.1
        problem += -0.1 <= p[ts, 4] <= 0.1
        problem += 0.9 <= p[ts, 5] <= 1.1

        problem += -0.1 <= p[ts, 3] + p[ts, 4] <= 0.1
        problem += -0.05 <= p[ts, 2] - p[ts, 5] <= 0.05

        # inclusion constraints
        for x, y in corners:
            problem += 0.0 <= p[ts, 0] + p[ts, 2] * x + p[ts, 3] * y <= h
            problem += 0.0 <= p[ts, 0] + p[ts, 4] * x + p[ts, 5] * y <= w

    problem.solve()
    print(f"info: problem solution is {pulp.LpStatus[problem.status]}")

    if problem.status != 1:
        return (problem.status, frame_transforms)

    best_transforms = np.full((number_of_frames, 3, 3), np.eye(3))
    return (problem.status, best_transforms)
