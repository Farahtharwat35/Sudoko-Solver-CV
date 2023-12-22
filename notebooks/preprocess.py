import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from hyperparameters import *


def display_img(img, cmap="gray"):
    plt.imshow(img, cmap=cmap)
    plt.show()


dir = "./Data set\CSE483 F23 Project Test Cases"
files = os.listdir(dir)


def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def angle_between_lines(line1, line2):
    l1x1, l1y1, l1x2, l1y2 = line1
    l2x1, l2y1, l2x2, l2y2 = line2
    a1 = np.rad2deg(np.arctan2(l1y2 - l1y1, l1x2 - l1x1))
    a2 = np.rad2deg(np.arctan2(l2y2 - l2y1, l2x2 - l2x1))
    return np.abs(a1 - a2)


def intersection_point(line1, line2):
    l1x1, l1y1, l1x2, l1y2 = line1
    l2x1, l2y1, l2x2, l2y2 = line2
    nx = (l1x1*l1y2-l1y1*l1x2)*(l2x1-l2x2)-(l2x1*l2y2-l2y1*l2x2)*(l1x1-l1x2)
    ny = (l1x1*l1y2-l1y1*l1x2)*(l2y1-l2y2)-(l2x1*l2y2-l2y1*l2x2)*(l1y1-l1y2)
    d = (l1x1-l1x2)*(l2y1-l2y2)-(l1y1-l1y2)*(l2x1-l2x2)
    px = int(nx / d)
    py = int(ny / d)
    return (px, py)


def point_on_line(point, line):
    def distance(pfrom, pto): return np.sqrt(
        (pfrom[0] - pto[0])**2 + (pfrom[1] - pto[1])**2)
    diff = distance(point, line[0:2]) + distance(point,
                                                 line[2:4]) - distance(line[0:2], line[2:4])
    # sus, this number should be close to zero, probably to allow broken lines
    return np.abs(diff) < point_on_line_error_pixels


for file in files:
    try:
        original_img = cv2.imread(os.path.join(
            dir, file), cv2.IMREAD_UNCHANGED)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

        # To check if the image is inverted
        if np.median(original_img) < 94 and original_img.std() > 48:
            original_img = cv2.bitwise_not(original_img)

        equalized_img = cv2.equalizeHist(original_img)
        equalized_img = cv2.medianBlur(equalized_img, 5)

        # print(f"{file} after eq: {equalized_img.mean()}")

        # cv2.imshow(file, cv2.resize(equalized_img, (400, 400)))

        binary_img = cv2.adaptiveThreshold(
            src=equalized_img,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=adaptive_threshold_block_size,
            C=adaptive_threshold_c
        )

        opening_kernel = np.ones((35, 35), np.uint8)
        eroded_img = cv2.morphologyEx(
            binary_img, cv2.MORPH_OPEN, opening_kernel)
        binary_img = binary_img - eroded_img

        # if binary_img.mean() > inversion_threshold:
        #     binary_img = cv2.bitwise_not(binary_img)

        # median_blurred_2 = cv2.medianBlur(binary_img, median_blur_widow_size)

        close_struct = np.ones((7, 7), np.uint8)
        open_struct = np.ones((7, 7), np.uint8)
        opened_img = cv2.morphologyEx(
            binary_img, cv2.MORPH_OPEN, open_struct, iterations=1)
        closed_img = cv2.morphologyEx(
            opened_img, cv2.MORPH_CLOSE, close_struct)

        minLineLength = max(
            original_img.shape[0], original_img.shape[1])/hough_min_line_length_ratio

        lines = cv2.HoughLinesP(
            closed_img,
            rho=hough_rho,
            theta=hough_theta,
            threshold=hough_threshold,
            minLineLength=minLineLength,
            maxLineGap=hough_max_line_gap
        )

        try:
            lines = lines.astype(np.int64)
        except Exception:
            print(f"couldn't find lines in {file}")
            continue

        binary_copy = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        lines_image = np.zeros_like(binary_img)

        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(binary_copy, (x1, y1), (x2, y2), (255, 0, 0), 1)

        img_height = len(original_img)
        img_width = len(original_img[0])

        intersections = []
        num_of_lines = len(lines[:, 0])
        # loop over each pair of lines
        for i in range(num_of_lines):
            for j in range(i+1, num_of_lines):
                line1 = lines[i, 0]
                line2 = lines[j, 0]
                if (line1 is line2):
                    continue
                a = angle_between_lines(line1, line2)
                # find the intersections between lines that are perpendicular on each other
                if (a < 80 or a > 100):
                    continue
                p = intersection_point(line1, line2)
                if (point_on_line(p, line1) or point_on_line(p, line2)):
                    # don't know why msha2lebhom but i assume that is the difference between the opencv and numpy
                    intersections.append(p[::-1])

        intersection_distance_threshold = 50

        intersections = list(set(intersections))
        filtered_points = set()
        to_be_removed = set()

        for i in range(len(intersections)):
            p1 = intersections[i]
            if p1 in to_be_removed:
                continue
            for j in range(i+1, len(intersections)):
                p2 = intersections[j]
                if p2 in to_be_removed:
                    continue
                if euclidean_distance(p1, p2) < intersection_distance_threshold:
                    to_be_removed.add(p2)

        intersections = list(set(intersections).difference(to_be_removed))

        p1 = sorted(intersections, key=lambda p: p[0] + p[1])[0]  # topleft
        p2 = sorted(intersections, key=lambda p: p[0] - p[1])[0]  # topright
        p3 = sorted(intersections, key=lambda p: p[0] + p[1])[-1]  # bottright
        p4 = sorted(intersections, key=lambda p: p[1] - p[0])[0]  # bottleft

        # print(p1, p2, p3, p4)
        # p1, p2, p3, p4 = find_largest_square(intersections, 70)
        # print(p1, p2, p3, p4)

        coords = np.int32([[p1[::-1], p2[::-1], p3[::-1], p4[::-1]]])

        corner_points_image = np.zeros_like(binary_copy)
        for coord in coords[0]:
            cv2.circle(corner_points_image,
                       (coord[0], coord[1]), 5, (255, 0, 0), 2)

        border_image = np.zeros_like(original_img, dtype=np.int32)
        border_image = cv2.polylines(
            border_image, coords, isClosed=True, color=(2550, 0, 0))

        y, x = binary_img.shape
        src_coords = np.float32([[0, 0], [x, 0], [x, y], [0, y]])
        dst_coords = np.float32([[p1[::-1], p2[::-1], p3[::-1], p4[::-1]]])
        img_gray_threshed_warped = cv2.warpPerspective(
            src=original_img,
            M=cv2.getPerspectiveTransform(dst_coords, src_coords),
            dsize=binary_img.shape[::-1]
        )

        M = img_gray_threshed_warped.shape[0] // 9
        N = img_gray_threshed_warped.shape[1] // 9
        number_tiles = []
        for i in range(9):
            number_tiles.append([])
            for j in range(9):
                tile = img_gray_threshed_warped[i*M:(i+1)*M, j*N:(j+1)*N]
                number_tiles[i].append(tile)
                path = os.path.join(f"./Data set/numbers/",
                                    f"{file}_{i}_{j}.jpg")
                cv2.imwrite(path, tile)

        # cv2.imshow(file, cv2.resize(img_gray_threshed_warped, (400, 400)))
    except Exception as e:
        print(f"error in {file}: {e}")

cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.show()
