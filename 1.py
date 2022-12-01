import cv2 as cv
import numpy as np


redBajo1 = np.array([0, 100, 20], np.uint8)
redAlto1 = np.array([8, 255, 255], np.uint8)
redBajo2 = np.array([175, 100, 20], np.uint8)
redAlto2 = np.array([179, 255, 255], np.uint8)
cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret == True:
        cv.imshow('frame', frame)
        frameHSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        maskRedl = cv.inRange(frameHSV, redBajo1, redAlto1)
        maskRed2 = cv.inRange(frameHSV, redBajo2, redAlto2)
        maskRed = cv.add(maskRedl, maskRed2)
        maskRedvis = cv.bitwise_and(frame, frame, mask=maskRed)
        cv.imshow('maskRedvis', maskRedvis)
        rgb = cv.cvtColor(maskRedvis, cv.COLOR_HSV2RGB)
        gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
        ret, thresh = cv.threshold(gray, 127, 255, 0)

        contornos, _ = cv.findContours(
            thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(frame, contornos, -1, (255, 0, 0), 3)
        cv.imshow('frame', frame)
        for c in contornos:
            area = cv.contourArea(c)
            if area > 3000:
                M = cv.moments(c)
                if (M["m00"] == 0):
                    M["m00"] = 1
                x = int(M["m10"]/M["m00"])
                y = int(M['m01']/M['m00'])
                cv.circle(frame, (2, 4), 7, (0, 255, 0), -1)

                font = cv.FONT_HERSHEY_SIMPLEX
                cv.putText(frame, '{},{}'.format(x, y),  (x+10, y),
                           font, 0.75, (0, 255, 0), 1, cv.LINE_AA)

                nuevoContorno = cv. convexHull(c)
                cv.drawContours(frame, [nuevoContorno], 0, (255, 0, 0), 3)
                cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('s'):
            break
cap.release()
cv.destroyAllWindows()                
