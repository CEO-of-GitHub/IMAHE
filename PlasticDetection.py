'''
 * Python program to use contours to count the objects in an image.
 *
 * usage: python Contours.py <filename> <threshold>
'''
import cv2, sys

# read command-line arguments
filename = sys.argv[1]
t = int(sys.argv[2])

# read original image
image = cv2.imread(filename = filename)

# create binary image
gray = cv2.cvtColor(src = image, code = cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(src = gray, 
    ksize = (5, 5), 
    sigmaX = 0)
(t, binary) = cv2.threshold(src = blur,
    thresh = t, 
    maxval = 255, 
    type = cv2.THRESH_BINARY)

contours, _ = cv2.findContours(image = binary, 
    mode = cv2.RETR_EXTERNAL,
    method = cv2.CHAIN_APPROX_SIMPLE)

print("Found %d objects." % len(contours))
for (i, c) in enumerate(contours):
    print("\tSize of contour %d: %d" % (i, len(c)))

cv2.drawContours(image = image, 
    contours = contours, 
    contourIdx = -1, 
    color = (0, 0, 255), 
    thickness = 5)

contours, hierarchy = cv2.findContours(image = binary, 
    mode = cv2.RETR_TREE,
    method = cv2.CHAIN_APPROX_SIMPLE)


cv2.imshow('image', image)
r=cv2.resize(image,(100,100))
cv2.imshow(r, image)
cv2.waitKey()


