import cv2
import numpy as np
from tqdm import tqdm
import os

def matchFeatures(input, output, nfeatures=1000):
    # I use ORB as feature detector
    orb = cv2.ORB_create(nfeatures=nfeatures, scoreType=cv2.ORB_FAST_SCORE)
    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(input, None)
    kp2, des2 = orb.detectAndCompute(output, None)
    # Finding matches with Brute Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    # Sort matches
    matches = sorted(matches, key=lambda x: x.distance)
    # take the matches and store them in array
    good = []
    for m in matches:
        good.append(m)
    # Find the source and destination points to use them in findHomography
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    """
    final_img = cv2.drawMatches(input, kp1,output, kp2, matches[:20], None)
    final_img = cv2.resize(final_img, (1000, 650))
    # Show the final image
    cv2.imshow("Matches1", final_img)
    cv2.waitKey()
    """

    # Find Homography Matrix through RANSAC
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return M

def warpImages(src, homography, dst):

    src = np.array(src)
    inputH = src.shape[0]
    inputW = src.shape[1]

    # Checking if image needs to be warped
    if homography == None:
        dst[H//2:H//2 + inputH, W//2:W//2 + inputW] = src
    else:
        # Calculating homography
        k = homography
        homography = np.eye(3)
        for i in range(len(k)):
            homography = k[i]

        # Finding bounding box
        points = np.array([[0, 0, 1], [inputW, inputH, 1], [inputW, 0, 1], [0, inputH, 1]]).T
        borders = (np.matmul(homography, points.reshape(3, -1)).reshape(points.shape))
        borders = borders / borders[-1]
        borders = (borders + np.array([W//2, H//2, 0])[:, np.newaxis]).astype(int)
        h_min, h_max = np.min(borders[1]), np.max(borders[1])
        w_min, w_max = np.min(borders[0]), np.max(borders[0])

        # Filling the bounding box in dst
        h_inv = np.linalg.inv(homography)
        for i in tqdm(range(h_min, h_max + 1)):
            for j in range(w_min, w_max + 1):

                if (0 <= i < H and 0 <= j < W):
                    # Calculating image coordinates for src
                    u, v = i - H//2, j - W//2
                    src_j, src_i, scale = np.matmul(h_inv , np.array([v, u, 1]))
                    src_i, src_j = int(src_i / scale), int(src_j / scale)

                    # Checking if coordinates lie within the image
                    if (0 <= src_i < inputH and 0 <= src_j < inputW):
                        dst[i, j] = src[src_i, src_j]


    # Creating a alpha mask of the transformed image
    mask = np.sum(dst, axis=2).astype(bool)
    return dst, mask


def laplacianPyramidBlend(images, masks, level=5):

    # Defining dictionaries for various pyramids
    g_pyramids = {}
    l_pyramids = {}
    W = images[0].shape[1]

    # Calculating pyramids for various images before hand
    for i in range(len(images)):
        # Gaussian Pyramids
        G = images[i].copy()
        g_pyramids[i] = [G]
        for k in range(level):
            G = cv2.pyrDown(G)
            g_pyramids[i].append(G)

        # Laplacian Pyramids
        l_pyramids[i] = [G]
        for j in range(len(g_pyramids[i])-2, -1, -1):
            G_up = cv2.pyrUp(G)
            G = g_pyramids[i][j]
            L = cv2.subtract(G, G_up)
            l_pyramids[i].append(L)

    # Blending Pyramids
    common_mask = masks[0].copy()
    common_pyramids = [l_pyramids[0][i].copy() for i in range(len(l_pyramids[0]))]

    ls_ = None
    # We take one image, blend it with our final image, and then repeat for n images
    for i in range(1, len(images)):
        y1, x1 = np.where(common_mask == 1)
        y2, x2 = np.where(masks[i] == 1)

        if np.max(x1) > np.max(x2):
            left_py = l_pyramids[i]
            right_py = common_pyramids

        else:
            left_py = common_pyramids
            right_py = l_pyramids[i]

        # To check if the two pictures need to be blended are overlapping or not
        mask_intersection = np.bitwise_and(common_mask, masks[i])

        if True in mask_intersection:
            # If images blend, we need to find the center of the overlap
            y, x = np.where(mask_intersection == 1)
            x_min, x_max = np.min(x), np.max(x)

            # We get the split point
            split = ((x_max-x_min)/2 + x_min)/W

            # Finally we add the pyramids
            LS = []
            for la, lb in zip(left_py, right_py):
                rows, cols, dpt = la.shape
                ls = np.hstack(
                    (la[:, 0:int(split*cols)], lb[:, int(split*cols):]))
                LS.append(ls)

        else:
            LS = []
            for la, lb in zip(left_py, right_py):
                rows, cols, dpt = la.shape
                ls = la + lb
                LS.append(ls)

        # Reconstructing the image
        ls_ = LS[0]
        for j in range(1, level+1):
            ls_ = cv2.pyrUp(ls_)
            ls_ = cv2.add(ls_, LS[j])

        # Preparing the common image for next image to be added
        common_image = ls_
        common_mask = np.sum(common_image.astype(bool), axis=2).astype(bool)
        common_pyramids = LS

    return ls_


# Path of img directory
# You should enter your directory or enter the input images to Images Array.
# We take as an array the input images which in the directory
IMG_DIR = r'input'
# Image Array
Images = []
valid_images = [".jpg", ".png", ".jpeg"]
# Shape of images H,W,C
A = 0
B = 0
C = 0
for f in os.listdir(IMG_DIR):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    img = cv2.imread(os.path.join(IMG_DIR, f))
    img = cv2.resize(img, (640,480))
    x,y,z = img.shape
    A = x
    B = y
    C = z
    Images.append(img)
# Image count
N = len(Images)
H, W, C = np.array((A, B, C))*[3, N, 1]
img_f = np.zeros((H, W, C))
img_outputs = []
masks = []
img, mask = warpImages(Images[1], None, img_f.copy())
img_outputs.append(img)
masks.append(mask)
left_H = []
right_H = []

for i in range(1, len(Images)//2+1):
    poc = matchFeatures(Images[N // 2 + i], Images[N // 2 + (i - 1)])
    right_H.append(poc)
    img1, mask = warpImages(Images[N // 2 + i], right_H[::-1], img_f.copy())
    img_outputs.append(img1)
    masks.append(mask)
    poc = matchFeatures(Images[N // 2 - i], Images[N // 2 - (i - 1)])
    left_H.append(poc)
    img1, mask = warpImages(Images[N // 2 - i], left_H[::-1],img_f.copy())
    img_outputs.append(img1)
    masks.append(mask)

uncropped = laplacianPyramidBlend(img_outputs, masks)
mask = np.sum(uncropped, axis=2).astype(bool)
# Finding appropriate bounding box
yy, xx = np.where(mask == 1)
x_min, x_max = np.min(xx), np.max(xx)
y_min, y_max = np.min(yy), np.max(yy)
# Croping and saving
final = uncropped[y_min:y_max, x_min:x_max]
cv2.imwrite("BlendedImage.jpg", final)
print("Process Successful")
