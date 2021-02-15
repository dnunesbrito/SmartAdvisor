# import the necessary packages
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ImageSegmentation import ImageSegmentation


def TailLightSegmentation(images: list) -> list:
    """Segments each image of the list separating the tail light of the cars
    Parameters: imges - A list of images with segmented image of vehicles
    Return: A list of positions of the outer bound of the tail light"""
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
            if cv2.contourArea(contours[i]) > 80:
                M = cv2.moments(contours[i])
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                height, width, channels = drawing.shape
                if height / 2 - 40 < cy < height / 2 + 40:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
                    # draw ith convex hull object
                    cv2.drawContours(drawing, hull, i, color, 1, 8)
                    x, y, w, h = cv2.boundingRect(hull[i])
                    cv2.drawMarker(drawing, tuple([x, y]), color=(0, 0, 255), markerType=cv2.MARKER_CROSS,
                                   thickness=2,
                                   markerSize=3)
                    cv2.rectangle(drawing, (x, y), (x + w, y + h), (155, 155, 0), 1)

        plt.imshow(drawing)
        plt.show()
