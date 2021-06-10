# import the necessary packages
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import rgb_to_hsv
from ImageSegmentation import ImageSegmentation


class FrontRearCarFeatures:
    """
    Class with a set of functions to detect cars on an image. The class have yet some function to detect front and
    tail light.

    Args:
        No arguments yet

    Attributes:
        No attributes yet.
    """
    def __init__(self):
        pass

    @staticmethod
    def cardetection(args: dict) -> list:
        """
        Static function which uses a Deep Neural Network to register some object of some types found in an image.

        Args:
            args (dict): A dictionary with the keys:
                prototxt: String with the path to the prototxt file It contains an image classification
                            or image segmentation model that is intended to be trained in Caffe.
                model: String with the path to the caffe model file. This is the model of the deep neural network.

        Returns:
            images (list): List of images with a rectangular form that contains the segmented cars found on input image.
        """
        # initialize the list of class labels MobileNet SSD was trained to detect, then generate a set of bounding box
        # colors for each class
        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                   "sofa", "train", "tvmonitor"]

        # load our serialized model from disk
        print("[INFO] loading model...")
        net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

        # load the input image and construct an input blob for the image by resizing to a fixed 300x300 pixels and
        # then normalizing it (note: normalization is done via the authors of the MobileNet SSD implementation)
        image = cv2.imread(args["image"])
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and predictions
        print("[INFO] computing object detections...")
        net.setInput(blob)
        detections = net.forward()

        images = []
        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
            if confidence > args["confidence"]:
                # extract the index of the class label from the `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # display the prediction
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                print("[INFO] {}".format(label))
                crop_img = image[startY:endY, startX:endX]
                images.append(crop_img)
        return images

    @staticmethod
    def SegmentationColorToLights(args: dict, dark_proportions: tuple, light_proportions: tuple):
        """
        Static function used to print two squares  with the upper and lower limit of the colors to detect some object of
        the color on image.

        Args:
            args (dict): dictionary of arguments no used.
            dark_proportions (tulpe): A tuple with the red, green, and blue proportions to the darker color.
            light_proportions (tuple): A tuple with the red, green, and blue proportions to the lighter color.

        Returns:
            Nothing, only shows the squares with the limit colors.
        """
        dark_color = np.dot(255, dark_proportions)
        light_color = np.dot(255, light_proportions)
        lo_square = np.full((10, 10, 3), light_color, dtype=np.uint8) / 255
        do_square = np.full((10, 10, 3), dark_color, dtype=np.uint8) / 255
        plt.subplot(1, 2, 1)
        plt.imshow(hsv_to_rgb(do_square))
        plt.subplot(1, 2, 2)
        plt.imshow(hsv_to_rgb(lo_square))
        plt.show()

