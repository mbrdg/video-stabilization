import numpy as np


def ComputeCentroidCoordinates(Input_Coords):
    # Compute the Normal matrix such that if the Input_Coords are
    # transformed by the Normal mat than they are centered around
    # the origin with an average point distance of sqrt(2) to the origin.

    # Input: 3xN Pseudo homogeneous coordinates of x
    # Outputs: Normal Matrix 3x3 transformation matrix,
    #          3xN normalized homogeneous coordinates

    if len(Input_Coords[0]) != 3:
        raise ValueError("Incorrect input dimensions")

    centroid = Input_Coords.mean(axis=0)

    distances = []

    for i in range(0, len(Input_Coords)):
        distance = np.sqrt(
            (
                np.power(Input_Coords[i][0] - centroid[0], 2)
                + np.power(Input_Coords[i][1] - centroid[1], 2)
                + np.power(Input_Coords[i][2] - centroid[2], 2)
            )
        )

        distances.append(distance)

    distances = np.asarray(distances)
    mean_dist = np.mean(distances)

    Norm_Mat = [
        [np.sqrt(2) / mean_dist, 0, -np.sqrt(2) / mean_dist * centroid[0]],
        [0, np.sqrt(2) / mean_dist, -np.sqrt(2) / mean_dist * centroid[1]],
        [0, 0, 1],
    ]

    Norm_Mat = np.asarray(Norm_Mat, dtype=float)
    Out_Coords = np.matmul(Norm_Mat, np.asarray(Input_Coords).T)

    return Out_Coords, Norm_Mat


def Homography(Coordinates_Image_1, Coordinates_Image_2):
    # Compute the Homography matrix (H) to perform a projective transformation

    # Input: 3xN Pseudo homogeneous coordinates of x
    # Outputs: Homography 3x3 transformation matrix,

    if len(Coordinates_Image_1) != len(Coordinates_Image_2):
        raise ValueError("Mismatch between number of points in the image pair")

    Num_Points = len(Coordinates_Image_1)
    Num_Cols_Homography_Mat = 8

    # Construct the design matrix and the observations
    A = np.zeros(shape=(2 * Num_Points, Num_Cols_Homography_Mat), dtype=float)
    B = np.zeros(shape=(2 * Num_Points, 1), dtype=float)

    counter = 0

    for i in range(0, 2 * Num_Points, 2):
        A[i, 0] = Coordinates_Image_1[counter, 0]
        A[i, 1] = Coordinates_Image_1[counter, 1]
        A[i, 2] = 1
        A[i, 3] = 0
        A[i, 4] = 0
        A[i, 5] = 0
        A[i, 6] = -(Coordinates_Image_1[counter, 0] * Coordinates_Image_2[counter, 0])
        A[i, 7] = -(Coordinates_Image_1[counter, 1] * Coordinates_Image_2[counter, 0])

        A[i + 1, 0] = 0
        A[i + 1, 1] = 0
        A[i + 1, 2] = 0
        A[i + 1, 3] = Coordinates_Image_1[counter, 0]
        A[i + 1, 4] = Coordinates_Image_1[counter, 1]
        A[i + 1, 5] = 1
        A[i + 1, 6] = -(
            Coordinates_Image_1[counter, 0] * Coordinates_Image_2[counter, 1]
        )
        A[i + 1, 7] = -(
            Coordinates_Image_1[counter, 1] * Coordinates_Image_2[counter, 1]
        )

        B[i, 0] = Coordinates_Image_2[counter, 0]
        B[i + 1, 0] = Coordinates_Image_2[counter, 1]

        counter = counter + 1

    # Compute the Least squares solution for Homography matrix
    H = np.matmul(
        np.matmul(np.linalg.inv(np.matmul(np.transpose(A), A)), np.transpose(A)), B
    )
    H = np.append(H, 1)

    H = np.reshape(H, (3, 3))

    return H


def HomographyTransformer(H_Matrix, X):
    # Transform the coordinates from one image to the other using homography matrix

    # Input: Homography 3x3 transformation matrix
    #        Vector of coordinates X
    # Output: Vector of transformed coordinates
    Tr_X = []

    for i in X:
        IC = np.asarray([i[0], i[1], 1])
        tr_X = np.matmul(H_Matrix, IC)

        X = tr_X[0] / tr_X[2]
        Y = tr_X[1] / tr_X[2]

        Tr_X.append([X, Y])

    return np.asarray(Tr_X)
