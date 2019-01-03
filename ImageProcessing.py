import cv2 as cv
import numpy as np

img = cv.imread('street.jpg', -1)
rows, cols, channel = img.shape
# 1
cv.imshow('image', img)

k = cv.waitKey(0)

# 2
if k == ord('2'):
    b, g, r = cv.split(img)
    cv.imshow('blue', r)
    cv.waitKey(0)
# 3
if k == ord('3'):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('grayscale', gray)
    cv.waitKey(0)
# 4
if k == ord('4'):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    grayAfterGaussian = cv.GaussianBlur(gray, (5, 5), 0)
    cv.imshow('gaussian applied', grayAfterGaussian)
    cv.waitKey(0)

# 5
if k == ord('5'):
    M = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 90, 1)
    rotatedimg = cv.warpAffine(img, M, (cols, rows))
    cv.imshow('rotated', rotatedimg)
    cv.waitKey(0)

# 6
if k == ord('6'):
    scaled = cv.resize(img, (int(rows / 2), cols), interpolation=cv.INTER_CUBIC)
    cv.imshow('scaled', scaled)
    cv.waitKey(0)
# 7
if k == ord('7'):
    edge = cv.Sobel(img, cv.CV_8U, 1, 1, ksize=5)
    cv.imshow('scaled', edge)
    cv.waitKey(0)

# 8
if k == ord('8'): # mn ba gaussian zdm ta behtar maloom she
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    grayAfterGaussian = cv.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv.threshold(grayAfterGaussian, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)
    cv.imshow('sure', sure_bg)

    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)
    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    markers = cv.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    cv.imshow('scaled', thresh)
    cv.waitKey(0)
