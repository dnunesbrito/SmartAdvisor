# USAGE
# python deep_learning_object_detection.py --image images/example_01.jpg \
#	--prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import rgb_to_hsv
from ImageSegmentation import ImageSegmentation
from FrontRearCarFeatures import FrontRearCarFeatures

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

#Detect all cars in the image
images = FrontRearCarFeatures.cardetection(args)

dark_red = np.dot(255, (0.1, 0.8, 1))

light_red = np.dot(255, (0, 0.4, 0.5))

FrontRearCarFeatures.SegmentationColorToLights(args, (0.1, 0.8, 1), (0, 0.4, 0.5))

for image in images:
    masked_image = ImageSegmentation(image)
    gray_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
    gray_image = cv2.medianBlur(gray_image, 3)
    th = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 3, 0)
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((5, 5), np.uint8)
    img_dilation = cv2.dilate(th, kernel, iterations=1)
    plt.imshow(img_dilation, "gray")
    plt.show()

    contours, hierarchy = cv2.findContours(img_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # create hull array for convex hull points
    hull = []
    # calculate points for each contour
    for i in range(len(contours)):
        # creating convex hull object for each contour
        hull.append(cv2.convexHull(contours[i], False))
    # create an empty black image
    drawing = np.zeros((th.shape[0], th.shape[1], 3), np.uint8)

    # draw contours and hull points
    for i in range(len(contours)):
        color_contours = (0, 255, 0)  # green - color for contours
        color = (255, 0, 0)  # blue - color for convex hull
        # draw ith contour
        if cv2.contourArea(contours[i]) > 100:
            M = cv2.moments(contours[i])
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            height, width, channels = drawing.shape
            if height/2 - 20 < cy < height/2 + 20:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
                # draw ith convex hull object
                cv2.drawContours(drawing, hull, i, color, 1, 8)
                x, y, w, h = cv2.boundingRect(hull[i])
                cv2.drawMarker(drawing, tuple([x, y]), color=(0, 0, 255), markerType=cv2.MARKER_CROSS, thickness=2,
                               markerSize=3)
                cv2.rectangle(drawing, (x, y), (x + w, y + h), (155, 155, 0), 1)

    plt.imshow(drawing)
    plt.show()
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # mask = cv2.inRange(hsv_image, light_red, dark_red)
    # result = cv2.bitwise_and(image, image, mask=mask)
    # img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    # ret, thresh_laplace = cv2.threshold(laplacian, 80, 255, cv2.THRESH_BINARY)
    # sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
    # ret, thresh_sobelx = cv2.threshold(sobelx, 80, 255, cv2.THRESH_BINARY)
    # sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)
    # ret, thresh_sobely = cv2.threshold(sobely, 80, 255, cv2.THRESH_BINARY)
    # plt.subplot(2, 2, 1)
    # plt.imshow(thresh_laplace, cmap='gray')
    # plt.subplot(2, 2, 2)
    # plt.imshow(img_gray, cmap='gray')
    # plt.subplot(2, 2, 3)
    # plt.imshow(thresh_sobelx, cmap='gray')
    # plt.subplot(2, 2, 4)
    # plt.imshow(thresh_sobely, cmap='gray')
    # plt.show()
