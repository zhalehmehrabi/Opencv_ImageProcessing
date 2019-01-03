import cv2 as cv
import numpy as np
import time

img = cv.imread('p.jpg', -1)
rows, cols, channel = img.shape
print(rows)
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
    scaled = cv.resize(img, (int(cols / 2), rows), interpolation=cv.INTER_AREA)
    cv.imshow('scaled', scaled)
    cv.waitKey(0)
# 7
if k == ord('7'):
    edge = cv.Sobel(img, cv.CV_8U, 1, 1, ksize=5)
    cv.imshow('scaled', edge)
    cv.waitKey(0)

# 8
if k == ord('8'):  # mn ba gaussian zdm ta behtar maloom she
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=3)
    # sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=5)
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)
    ret, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    cv.imshow('segmented', img)
    cv.waitKey(0)

# 9
if k == ord('9'):
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1, 5)
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    cv.imshow('img', img)
# 10
if k == ord('0'):
    i = 0
    cap = cv.VideoCapture('output.avi')

    while (cap.isOpened()):
        ret, frame = cap.read()
        #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow('frame', frame)
        cv.waitKey(500)
        if i == 5:
            break
        i += 1
    cap.release()
    cv.destroyAllWindows()

if k ==ord('r'):
    cap = cv.VideoCapture(0)
    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            frame = cv.flip(frame, 0)
            # write the flipped frame
            rows, cols, channel = frame.shape
            M = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 180, 1)
            dst = cv.warpAffine(frame, M, (cols, rows))
            out.write(dst)
            cv.imshow('frame', dst)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    # Release everything if job is finished
    cap.release()
    out.release()
    cv.destroyAllWindows()