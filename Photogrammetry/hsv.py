import cv2
import numpy as np

# Read image
img = cv2.imread("DJI_0086.JPG")

# Resize image
resized_img = cv2.resize(img, (600,600), interpolation = cv2.INTER_AREA)

# Find the hsv mask and isolate the white regions.
sensitivity = 35
hsv = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
lower_white = np.array([0,0,255-sensitivity], dtype=np.uint8)
upper_white = np.array([255,sensitivity,255], dtype=np.uint8)
mask = cv2.inRange(hsv, lower_white, upper_white)
res = cv2.bitwise_and(resized_img, resized_img, mask=mask)

# Convert to gray scale
res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

# Defining the filters. Four filters for four different orientations of the GCP
b1 = np.array((
	[1, 1],
	[0, 1]), dtype = "int")
b2 = np.array((
	[1, 1],
	[1, 0]), dtype = "int")
b3 = np.array((
	[0, 1],
	[1, 1]), dtype = "int")
b4 = np.array((
	[1, 0],
	[1, 1]), dtype = "int")

# Execution of the HITMISS algorithm
output_image = cv2.morphologyEx(res, cv2.MORPH_HITMISS, b2)

flag, thresh = cv2.threshold(output_image, 130, 255, cv2.THRESH_BINARY)

# Dilating the image as GCP's get distorted after HITMISS
dilate = cv2.dilate(thresh, (1,1), iterations=1)
cv2.imshow("Image", output_image)

# Using Canny Edge Detector to detect edges and Contour detection for drawing Bboxe's
edges = cv2.Canny(dilate, 0, 200, 255)
im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Drawing contours to identify the coordinates. 
# Prints the x, y, w, h -coordinate of all the probable locations.
for c in contours:
	x,y,w,h = cv2.boundingRect(c)
	if h<=1 or w<=1:
		cv2.rectangle(resized_img, (x,y), (x+w, y+h), (0,0,0),1)
		cv2.imshow("Image", resized_img)
		print x, y, w, hierarchy

cv2.waitKey(0);
cv2.destroyAllWindows()
