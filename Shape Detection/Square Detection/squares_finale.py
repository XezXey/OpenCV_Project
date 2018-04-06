#!/usr/bin/env python

'''
Simple "Square Detector" program.

Loads several images sequentially and tries to find squares in each image.
'''
####################
USE_CAM = True
####################


# Python 2/3 compatibility
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import math
import numpy as np
import cv2 as cv


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img):
    img = cv.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv.split(img):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv.Canny(gray, 0, 50, apertureSize=5)
                bin = cv.dilate(bin, None)
            else:
                _retval, bin = cv.threshold(gray, thrs, 255, cv.THRESH_BINARY)
            bin, contours, _hierarchy = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv.arcLength(cnt, True)
                cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv.contourArea(cnt) > 1000 and cv.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares

def compare_increment(initial_square, final_square):
    initial_square_attr = {"area" : 0, "width" : 0, "height" : 0}
    final_square_attr = {"area" : 0, "width" : 0, "height" : 0}
    different_square_attr = {"area" : 0, "width" : 0, "height" : 0}
    
    #Calculate initial square attributes
    initial_square_attr["width"] = math.sqrt(pow(initial_square[0][0] - initial_square[2][0], 2) + pow(initial_square[0][1] - initial_square[2][1], 2))
    initial_square_attr["height"] = math.sqrt(pow(initial_square[0][0] - initial_square[1][0], 2) + pow(initial_square[0][1] - initial_square[1][1], 2))
    initial_square_attr["area"] = initial_square_attr["width"] * initial_square_attr["height"]
    print("************************************************", end = "\n")
    print("Initial square : \n" + "Area : " + str(initial_square_attr["area"])
            + "\nWidth : " + str(initial_square_attr["width"])
            + "\nHeight : " + str(initial_square_attr["height"]), end = "\n")
    print("************************************************", end = "\n")


    #Calculate final square attributes
    final_square_attr["width"] = math.sqrt(pow(final_square[0][0] - final_square[2][0], 2) + pow(final_square[0][1] - final_square[2][1], 2))
    final_square_attr["height"] = math.sqrt(pow(final_square[0][0] - final_square[1][0], 2) + pow(final_square[0][1] - final_square[1][1], 2))
    final_square_attr["area"] = final_square_attr["width"] * final_square_attr["height"]
    print("************************************************", end = "\n")
    print("Final square : \n" + "Area : " + str(final_square_attr["area"])
            + "\nWidth : " + str(final_square_attr["width"])
            + "\nHeight : " + str(final_square_attr["height"]), end = "\n")
    print("************************************************", end = "\n")

    #Calculate difference
    different_square_attr["width"] = final_square_attr["width"] - initial_square_attr["width"]
    different_square_attr["height"] = final_square_attr["height"] - initial_square_attr["height"]
    different_square_attr["area"] = final_square_attr["area"] - initial_square_attr["area"]
    print("************************************************", end = "\n")
    print("Difference : \n" + "Area : " + str(different_square_attr["area"])
            + "\nWidth : " + str(different_square_attr["width"])
            + "\nHeight : " + str(different_square_attr["height"]), end = "\n")
    print("************************************************", end = "\n")

    

def find_smallest_square(squares, img):
    original_height, original_width, original_channels = img.shape
    min_area = original_height * original_width
    area = 0
    min_area_index = 0
    #The point of each square is not order but we can know that point[0] opposite to point[3] also point[1] and point[2]
    for index, each_square in enumerate(squares):
        """
        cv.circle(img, tuple(each_square[0]), 20, (0, 0, 0))    #BLACK
        cv.circle(img, tuple(each_square[1]), 20, (0, 255, 0))  #GREEN
        cv.circle(img, tuple(each_square[2]), 20, (255, 0, 0))    #RED
        cv.circle(img, tuple(each_square[3]), 20, (0, 0, 255))  #BLUE
        """
        height = math.sqrt(pow(each_square[0][0] - each_square[1][0], 2) + pow(each_square[0][1] - each_square[1][1], 2))
        width = math.sqrt(pow(each_square[0][0] - each_square[2][0], 2) + pow(each_square[0][1] - each_square[2][1], 2))
        area = int(height * width)
        #print("Area = " + str(area) + ", Min Area = " + str(min_area))
        if (area < min_area) and (area != original_height - 1) * (original_width - 1):
            #print("Min Area : " + str(min_area))
            min_area = area
            min_area_index = index
            print("INDEX : " + str(index))
    #print(squares[min_area_index])
    if squares[min_area_index] is None:
        print("NULL")
    return squares[min_area_index]
        
if __name__ == '__main__':
    from glob import glob
    if not USE_CAM:
        for fn in glob("./*.jpg"):
            img = cv.imread(fn)
            img = cv.resize(img, (900, 900))    # Resizing image may change if cant detect the square
            squares = find_squares(img)
            #Squares is list of each square and each square contain 4 point
            squares = find_smallest_square(squares, img)
            #squares = np.array(squares).reshape((-1, 1, 2)).astype(np.int32)
            cv.drawContours(img, [squares.astype(np.int32)], -1, (0, 255, 0), 3)
            cv.circle(img, tuple(squares[0]), 20, (0, 0, 0))    #BLACK
            cv.circle(img, tuple(squares[1]), 20, (0, 255, 0))  #GREEN
            cv.circle(img, tuple(squares[2]), 20, (255, 0, 0))    #RED
            cv.circle(img, tuple(squares[3]), 20, (0, 0, 255))  #BLUE
            cv.imshow('squares', img)
            ch = cv.waitKey()
            if ch == 115:   #Press S to save (ASCII)
                initial_square = squares
                print("Init an initial square!!!")
            elif ch == 83:
                final_square = squares
                print("Init a final square!!!")
                compare_increment(initial_square, final_square)
            elif ch == 113:
                print("EXIT...")
                break
                
    else:
        cap = cv.VideoCapture("http://192.168.111.16:8080/video")
        while(True):
            ret, frame = cap.read()
            frame = cv.resize(frame, (900, 900)) 
            squares = find_squares(frame)
            if not squares :
                print("NULL")
                continue
            squares = find_smallest_square(squares, frame)
            cv.drawContours(frame, [squares.astype(np.int32)], -1, (0, 255, 0), 3)
            cv.circle(frame, tuple(squares[0]), 20, (0, 0, 0))    #BLACK
            cv.circle(frame, tuple(squares[1]), 20, (0, 255, 0))  #GREEN
            cv.circle(frame, tuple(squares[2]), 20, (255, 0, 0))    #RED
            cv.circle(frame, tuple(squares[3]), 20, (0, 0, 255))  #BLUE
            cv.imshow("frame", frame)
            h = cv.waitKey()
            if ch == 115:   #Press S to save (ASCII)
                initial_square = squares
                print("Init an initial square!!!")
            elif ch == 83:
                final_square = squares
                print("Init a final square!!!")
                compare_increment(initial_square, final_square)
            elif ch == 113:
                print("EXIT...")
                break
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    
    cv.destroyAllWindows()
