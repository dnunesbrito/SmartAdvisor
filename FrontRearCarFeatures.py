# USAGE
# python deep_learning_object_detection.py --image images/example_01.jpg \
#	--prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
import numpy as np
import cv2
import matplotlib.pyplot as plt


class FrontRearCarFeatures:

    def __init__(self):
        pass

    @staticmethod
    def CarDetection(args):
        # initialize the list of class labels MobileNet SSD was trained to
        # detect, then generate a set of bounding box colors for each class
        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                   "sofa", "train", "tvmonitor"]
        # COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

        # load our serialized model from disk
        print("[INFO] loading model...")
        net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

        # load the input image and construct an input blob for the image
        # by resizing to a fixed 300x300 pixels and then normalizing it
        # (note: normalization is done via the authors of the MobileNet SSD
        # implementation)
        image = cv2.imread(args["image"])
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        print("[INFO] computing object detections...")
        net.setInput(blob)
        detections = net.forward()

        images = []
        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > args["confidence"]:
                # extract the index of the class label from the `detections`,
                # then compute the (x, y)-coordinates of the bounding box for
                # the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # display the prediction
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                print("[INFO] {}".format(label))
                # cv2.rectangle(image, (startX, startY), (endX, endY),
                #              COLORS[idx], 2)
                # y = startY - 15 if startY - 15 > 15 else startY + 15
                # cv2.putText(image, label, (startX, y),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                crop_img = image[startY:endY, startX:endX]
                images.append(crop_img)
        return images

    def TailLightSegmentation(self, images: list) -> list:
        # implement return of a list of positions of the taillights to each image
        for image in images:
            masked_image = self.ImageSegmentation(image)
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
                    if height / 2 - 20 < cy < height / 2 + 20:
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

    def ImageSegmentation(self, image):

        # convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # reshape the image to a 2D array of pixels and 3 color values (RGB)
        pixel_values = image.reshape((-1, 3))
        # convert to float
        pixel_values = np.float32(pixel_values)

        # print(pixel_values.shape)

        # define stopping criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

        # number of clusters (K)
        k = 8
        _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        dark_red = (0.1 * 255, 0.8 * 255, 1 * 255)
        light_red = (0 * 255, 0.4 * 255, 0.5 * 255)
        on_labels = []
        cont_labels = 0
        for center in centers:
            lo_square = np.full((10, 10, 3), center, dtype=np.uint8)
            lo_square_hsv = cv2.cvtColor(lo_square, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(lo_square_hsv, light_red, dark_red)
            value = mask[0][0]
            if value > 0:
                on_labels.append(cont_labels)
            cont_labels = cont_labels + 1

        # convert back to 8 bit values
        centers = np.uint8(centers)

        # flatten the labels array
        labels = labels.flatten()

        # convert all pixels to the color of the centroids
        segmented_image = centers[labels.flatten()]

        # reshape back to the original image dimension
        segmented_image = segmented_image.reshape(image.shape)

        # disable only the cluster number 2 (turn the pixel into black)
        masked_image = np.copy(image)
        # convert to the shape of a vector of pixel values
        masked_image = masked_image.reshape((-1, 3))
        # color (i.e cluster) to disable
        # masked_image[labels == cluster] = [0, 0, 0]
        for cluster_off in range(0, k):
            if cluster_off not in on_labels:
                masked_image[labels == cluster_off] = [0, 0, 0]

        # convert back to original shape
        masked_image = masked_image.reshape(image.shape)

        return masked_image
