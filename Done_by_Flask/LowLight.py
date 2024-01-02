#导包
import cv2
import numpy as np
import matplotlib.pyplot as plt

#直方图均衡化
def EqualHist():
    image = cv2.imread("images/00000.JPG")
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalHist = cv2.equalizeHist(grayImage,None)
    hist = cv2.calcHist(equalHist,[0],None,[256],[0,255])
    plt.title("EqualHist of GrayImage")
    plt.xlabel("Amount of Pixel")
    plt.ylabel("Gray Scale")
    plt.plot(hist)
    #plt.savefig("Equalhist")
    # plt.show()

    down_width = 300
    down_height = 300
    down_points = (down_width, down_height)
    equalHist = cv2.resize(equalHist, down_points, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("EqualHist",equalHist)
    cv2.imwrite("equalHist.jpg",equalHist)
    cv2.waitKey(0)

EqualHist()
