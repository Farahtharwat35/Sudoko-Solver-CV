import numpy as np

# Hyper parameters
# Inversion
inversion_threshold = 80

# Thresholding
adaptive_threshold_block_size = 31
adaptive_threshold_c = 10

# Median Blur
median_blur_widow_size = 7

# Hough
hough_min_line_length_ratio = 2.9
hough_rho = 1
hough_theta = np.pi/360
hough_threshold = 307
hough_max_line_gap = 55

# Point on line
point_on_line_error_pixels = 500

# Intersections
intersections_error_pixels = 10

# Test images padding
test_image_padding = 50
