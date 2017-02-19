#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2


# built-in modules
import os
import sys
import math

from sklearn.externals import joblib
from skimage.feature import hog


from digits import *

def getLine():
    cap = cv2.VideoCapture('video-5.avi')
    maxX1 = 0
    maxX2 = 0
    maxY1 = 0
    maxY2 = 0
    maxDistance = 0
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=7)
            minLineLength = 10
            maxLineGap = 0
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
            for x1, y1, x2, y2 in lines[0]:
                distance = math.sqrt(math.pow(x2 -x1, 2) + math.pow(y2 -y1, 2))
                if distance > maxDistance:
                    maxDistance = distance
                    maxX1 = x1
                    maxX2 = x2
                    maxY1 = y1
                    maxY2 = y2
        except:
            return maxX1, maxX2, maxY1, maxY2

def main():

    suma = 0
    cap = cv2.VideoCapture('video-5.avi')

    clf = joblib.load("digits_cls.pkl")

    [lx1, lx2, ly1, ly2] = getLine()
    lineValues = {}
    lastNumber = -1

    for i in range(lx1, lx2):
        lineValues[i] = (((ly2 - ly1) * 1.0) / (lx2 - lx1)) * (i - lx1) + ly1

    while cap.isOpened():
        try:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
        except:
            return

        ret, treshold = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
        _, ctrs, hier = cv2.findContours(treshold.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        rects = []

        for ctr in ctrs:
            [x, y, w, h] = cv2.boundingRect(ctr)
            if h < 15:
                continue
            rects.append(cv2.boundingRect(ctr))

        cv2.line(frame, (lx1, ly1), (lx2, ly2), (0, 0, 255), 2)

        for rect in rects:
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 1)
            l = int(rect[3] * 1.6)
            pt1 = int(rect[1] + rect[3] // 2 - l // 2)
            pt2 = int(rect[0] + rect[2] // 2 - l // 2)
            roi = treshold[pt1:pt1 + l, pt2:pt2 + l]
            try:
                roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                roi = cv2.dilate(roi, (3, 3))
            except:
                continue

            roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
            nbr = clf.predict(np.array([roi_hog_fd], 'float64'))

            number = int(nbr[0])

            if rect[0] in lineValues.keys():
                if rect[1] > lineValues[rect[0]] - 3 and rect[1] < lineValues[rect[0]] + 3:
                    if lastNumber == -1:
                        lastNumber = number
                        suma = suma + number
                    if number != lastNumber:
                        lastNumber = -1


            cv2.putText(frame, str(int(nbr[0])), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 149, 255), 1)
            cv2.putText(frame, 'Suma: {0}'.format(suma), (10,50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 149, 255), 1)

        cv2.imshow('frame', frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
