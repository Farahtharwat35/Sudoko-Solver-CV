import cv2
import numpy as np

# Read the image
image = cv2.imread('your_image_path.jpg')  # Replace 'your_image_path.jpg' with the path to your image

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding or any other preprocessing if needed
# For example, to apply a simple binary threshold:
_, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
contour_image = np.copy(image)
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)  # -1 to draw all contours, (0, 255, 0) for color, 2 for thickness

# Display the original image with contours
cv2.imshow('Image with Contours', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
