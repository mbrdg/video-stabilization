import cv2
import matplotlib.pyplot as plt
import numpy as np


# Images must come in grayscale
def get_matching_features(img1_name, img2_name):
    img1 = cv2.imread(img1_name, 0)  # queryImage
    img2 = cv2.imread(img2_name, 0)  # trainImage
    # Initiate SIFT detector

    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    #  FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # Need to draw only good matches, so create a mask
    matching_coords = [m for m, n in matches if m.distance < 0.7 * n.distance]
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matching_coords]).reshape(
        -1, 1, 2
    )
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matching_coords]).reshape(
        -1, 1, 2
    )

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        matchesMask=matchesMask,
        flags=cv2.DrawMatchesFlags_DEFAULT,
    )
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matching_coords, None, **draw_params)
    plt.imshow(
        img3,
    ), plt.show()
    # Returns the pairs of matching coordinates
    return [(kp1[m.queryIdx].pt, kp2[m.trainIdx].pt) for m in matching_coords]


if __name__ == "__main__":
    get_matching_features("sample1.jpg", "sample2.jpg")
