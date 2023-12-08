import copy

import cv2
import numpy as np

# read img convert it to grayscale
colored_img = cv2.imread('D:/UNI/sems/2023 fall/Computer Vision/Project/data set/sodoku3.jpg')
original_img = cv2.cvtColor(colored_img, cv2.COLOR_RGB2GRAY)
cv2.imshow('sodoku pic', original_img)

# Apply Gaussian blur
kernel_size = (5, 5)  # Define the size of the Gaussian kernel (should be odd numbers)
sigma_x = 0  # Standard deviation in X direction (0 implies calculated from kernel size)
blurred_image = cv2.GaussianBlur(original_img, kernel_size, sigma_x)
cv2.imshow('Blurred Image', blurred_image)

# Global thresholding using OTSU method
ret, thresh_otsu = cv2.threshold(original_img, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.namedWindow('OTSU method', cv2.WINDOW_NORMAL)
cv2.imshow('OTSU method', thresh_otsu)
# Global thresholding is bad

# local mean thresholding
thresh_org_local_mean = cv2.adaptiveThreshold(original_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                              cv2.THRESH_BINARY, 11, 2)
cv2.imshow('local org mean', thresh_org_local_mean)

thresh_blured_local_mean = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                 cv2.THRESH_BINARY, 11, 2)
cv2.imshow('local blured mean', thresh_blured_local_mean)

# local gaus thresholding
thresh_org_local_gaus = cv2.adaptiveThreshold(original_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25,
                                              2)
cv2.imshow('local org gaus', thresh_blured_local_mean)
thresh_blured_local_gaus = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                 25, 2)
cv2.imshow('local blured gaus', thresh_blured_local_mean)

# Median filter
median_TOLM = cv2.medianBlur(thresh_org_local_mean, 3)
for i in range(0):
    median_TOLM = cv2.medianBlur(median_TOLM, 3)
cv2.imshow('median_TOLM', median_TOLM)

_, inverted_thresh = cv2.threshold(median_TOLM, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow('inverted_median_TOLM', inverted_thresh)











# erode the img ro remove noise
radius = 1  # Radius of the circle
circle_SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))

# Perform erosion using the circular structuring element
eroded_image = cv2.erode(inverted_thresh, circle_SE)
for i in range(0):
    eroded_image = cv2.erode(eroded_image, circle_SE)
cv2.imshow('eroded_image', eroded_image)

median_filtered = cv2.medianBlur(eroded_image, 3)
cv2.imshow('median', median_filtered)

# Apply Canny edge detection
edges = cv2.Canny(median_filtered, 50, 150, apertureSize=3)

# Apply Hough Line Transform
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

cv2.imshow('edges', edges)

colored_img_copy = copy.deepcopy(colored_img)

# Draw detected lines on the original image
if lines is not None:
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(colored_img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Display the result
cv2.imshow('Detected Lines', colored_img_copy)


dilated_img = cv2.dilate(median_filtered, circle_SE)
cv2.imshow('dilated', dilated_img)
# --------------------------------------------
cv2.waitKey(0)
cv2.destroyAllWindows()
# -----------------------------------------



dst = cv2.cornerHarris(edges, blockSize=5, ksize=3, k=0.22)
cv2.imshow('dst', dst)

# Dilate the corners to make them more visible
dst = cv2.dilate(dst, None)
cv2.imshow('dst2', dst)
# Threshold the corners to consider only strong corners
threshold = 0.01 * dst.max()
corner_image = np.copy(colored_img)
corner_image[dst > threshold * dst] = [0, 0, 255]

# Display the image with detected corners
cv2.namedWindow('Corners Detected', cv2.WINDOW_NORMAL)
cv2.imshow('Corners Detected', corner_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
