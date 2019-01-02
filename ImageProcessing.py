import cv2 as cv
import numpy as np
img = cv.imread('download.jpg', -1)
#cv.imshow('image', img)
while 1:
    k = cv.waitKey(0)
    # 1
    if k == ord('1'):
        print("slm")
        cv.imshow('image', img)
        cv.waitKey(5)
    # 2
    if k == ord('2'):
        b, g, r = cv.split(img)
        cv.imshow('blue', r)
        cv.waitKey(0)
    # 3
    if k == ord('3'):
        gray = cv.cvtColor(img , cv.COLOR_BGR2GRAY)
        cv.imshow('grayscale', gray)
        cv.waitKey(0)
    if k == ord('s'):
        break
cv.waitKey(0)


