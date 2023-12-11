import cv2
import numpy as np
from matplotlib import pyplot as plt

def display_img(img, cmap="gray"):
    plt.imshow(img, cmap=cmap)
    plt.show()
    
import os

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
    def distance(pfrom, pto): return np.sqrt((pfrom[0] - pto[0])**2 + (pfrom[1] - pto[1])**2)
    diff = distance(point, line[0:2]) + distance(point, line[2:4]) - distance(line[0:2], line[2:4])
    return np.abs(diff) < 100 # sus, this number should be close to zero, probably to allow broken lines


for file in files:
    original_img = cv2.imread(os.path.join(dir, file), cv2.IMREAD_UNCHANGED)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        
        
    if original_img.mean() < 128: 
        original_img = cv2.bitwise_not(original_img)
        

    binary_img = cv2.adaptiveThreshold(
        src=original_img,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=111,
        C=5
    ) 
    
    binary_img = cv2.medianBlur(binary_img, 5)
    
    
    minLineLength = min(original_img.shape[0],original_img.shape[1])/2

    lines = cv2.HoughLinesP(
        binary_img,
        rho=1,
        theta=np.pi/360,
        threshold=150,
        minLineLength=minLineLength,
        maxLineGap=71
    )

    try:
        lines = lines.astype(np.int64)
    except Exception:
        print(f"couldn't find lines in {file}")
        continue
    

    binary_copy = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    lines_image = np.zeros_like(binary_img)

    for x1, y1, x2, y2 in lines[:,0]: cv2.line(lines_image,(x1,y1),(x2,y2),(255,0,0),1)
    for x1, y1, x2, y2 in lines[:,0]: cv2.line(binary_copy,(x1,y1),(x2,y2),(255,0,0),1)
    

    intersections_image = np.zeros_like(original_img, dtype=np.uint16)

    img_height = len(original_img)
    img_width = len(original_img[0])

    intersections = []
    num_of_lines = len(lines[:,0])
    # loop over each pair of lines
    for i in range(num_of_lines):
        for j in range(i+1, num_of_lines):
            line1 = lines[i,0]
            line2 = lines[j,0]
            if (line1 is line2): continue
            a = angle_between_lines(line1, line2)
            # find the intersections between lines that are perpendicular on each other
            if (a < 80 or a > 100): continue
            p = intersection_point(line1, line2)
            if point_on_line(p, line1) and point_on_line(p, line2) and p[0] < img_height and p[1] < img_width:
                # don't know why msha2lebhom but i assume that is the difference between the opencv and numpy
                intersections_image[p[::-1]] = 5000
                intersections.append(p[::-1])

    p1 = sorted(intersections, key = lambda p: p[0] + p[1])[0] # topleft
    p2 = sorted(intersections, key = lambda p: p[0] - p[1])[0] # topright
    p3 = sorted(intersections, key = lambda p: p[0] + p[1])[-1] # bottright
    p4 = sorted(intersections, key = lambda p: p[1] - p[0])[0] # bottleft

    # print(p1, p2, p3, p4)
    # p1, p2, p3, p4 = find_largest_square(intersections, 70)
    # print(p1, p2, p3, p4)

    coords = np.int32([[p1[::-1], p2[::-1], p3[::-1], p4[::-1]]])


    corner_points_image = np.zeros_like(binary_copy)
    for coord in coords[0] : cv2.circle(corner_points_image, (coord[0], coord[1]), 5, (255, 0, 0), 2)

    border_image = np.zeros_like(original_img, dtype = np.int32)
    border_image = cv2.polylines(border_image, coords, isClosed=True, color=(2550, 0, 0))
    
    y, x = binary_img.shape
    src_coords = np.float32([[0,0], [x,0], [x,y], [0,y]])
    dst_coords = np.float32([[p1[::-1], p2[::-1], p3[::-1], p4[::-1]]])
    img_gray_threshed_warped = cv2.warpPerspective(
        src=binary_img,
        M=cv2.getPerspectiveTransform(dst_coords, src_coords),
        dsize=binary_img.shape[::-1]
    )
    plt.imshow(img_gray_threshed_warped, cmap="gray")
    plt.title(file)
    plt.show()