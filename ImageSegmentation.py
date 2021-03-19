import cv2
import numpy as np
import matplotlib.pyplot as plt


def ImageSegmentation(image):

    #image = cv2.imread("demonstration-image.png")
    # convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))
    # convert to float
    pixel_values = np.float32(pixel_values)

    #print(pixel_values.shape)

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
            #plt.imshow(lo_square)
            #plt.show()
        cont_labels = cont_labels + 1

    # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()

    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]

    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)
    # show the image
    #plt.imshow(segmented_image)
    #plt.show()

    # disable only the cluster number 2 (turn the pixel into black)
    masked_image = np.copy(image)
    # convert to the shape of a vector of pixel values
    masked_image = masked_image.reshape((-1, 3))
    # color (i.e cluster) to disable
    #masked_image[labels == cluster] = [0, 0, 0]
    for cluster_off in range(0, k):
        if cluster_off not in on_labels:
            masked_image[labels == cluster_off] = [0, 0, 0]

    # convert back to original shape
    masked_image = masked_image.reshape(image.shape)

    return masked_image
