# USAGE
# python deep_learning_object_detection.py --image images/example_01.jpg \
# --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import rgb_to_hsv
from ImageSegmentation import ImageSegmentation
from CarDetection import *
from TailLightSegmentation import *


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

# Detect all cars in the image
images = CarDetection(args)

dark_red = np.dot(255, (0.1, 0.8, 1))
light_red = np.dot(255, (0, 0.4, 0.5))

TailLightSegmentation(images)