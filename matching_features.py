import cv2
import matplotlib.pyplot as plt


# Images must come in grayscale
def get_matching_features(img1_name, img2_name):
    img1 = cv2.imread(img1_name, cv2.IMREAD_GRAYSCALE)  # queryImage
    img2 = cv2.imread(img2_name, cv2.IMREAD_GRAYSCALE)  # trainImage
    # Initiate SIFT detector

    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    #  FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for _ in range(len(matches))]
    matching_coords = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
            matching_coords.append(m)
    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        matchesMask=matchesMask,
        flags=cv2.DrawMatchesFlags_DEFAULT,
    )
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    plt.imshow(
        img3,
    ), plt.show()
    # Returns the pairs of matching coordinates
    return [(kp1[m.queryIdx].pt, kp2[m.trainIdx].pt) for m in matching_coords]
