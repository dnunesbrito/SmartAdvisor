# import the necessary packages
import numpy as np
import cv2


def CarDetection(args: dict) -> list:
    """Detects a vehicle in an image:
    Parameter: args -> A dictionary with the fields:
                prototxt: String with the path to the prototxt file It contains an image classification
                            or image segmentation model that is intended to be trained in Caffe.
                model: String with the path to the caffe model file. This is the model of the deep neural network.
                image: String with file address to image file to detect the car
                confidence: The confidence to consider a vehicle as a car
    Return: An image list with image of segmented cars    """

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
