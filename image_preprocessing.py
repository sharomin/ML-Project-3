import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
import pandas as pd


def preprocess(images_train, images_test):

    largestsize_train = []
    largestsize_test = []

    for i, image in enumerate(images_train):
        image = images_train[i].astype('uint8')
        ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        contourImage, contours, hierarchy = cv2.findContours(thresh, 1, 2)
        largest_areas = sorted(contours,
                               key=lambda cont: cv2.minAreaRect(cont)[
                                   1][0]*cv2.minAreaRect(cont)[1][1])
        cnt = largest_areas[-1]
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        poly = np.array([box], dtype=np.int32)
        mask = np.zeros(image.shape, np.uint8)
        cv2.fillPoly(mask, poly, 255)
        preprocessed_image = cv2.bitwise_and(image, mask)
        largestsize_train.append(preprocessed_image)

    for i, image in enumerate(images_test):
        image = images_test[i].astype('uint8')
        ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        contourImage, contours, hierarchy = cv2.findContours(thresh, 1, 2)
        largest_areas = sorted(contours, key=lambda cont: cv2.minAreaRect(cont)[
                               1][0]*cv2.minAreaRect(cont)[1][1])
        cnt = largest_areas[-1]
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        poly = np.array([box], dtype=np.int32)
        mask = np.zeros(image.shape, np.uint8)
        cv2.fillPoly(mask, poly, 255)
        preprocessed_image = cv2.bitwise_and(image, mask)
        largestsize_test.append(preprocessed_image)

    return largestsize_train, largestsize_test


x_train_images = pd.read_pickle(
    'drive/My Drive/ML_miniproject_3/train_images.pkl')
x_test_images = pd.read_pickle(
    'drive/My Drive/ML_miniproject_3/test_images.pkl')
y_train_labels = pd.read_csv(
    'drive/My Drive/ML_miniproject_3//train_labels.csv')

filtered_datatrain = x_train_images
filtered_datatrain[filtered_datatrain < 240] = 0
filtered_datatrain[filtered_datatrain >= 240] = 255
filtered_datatrain = filtered_datatrain.reshape(-1, 64, 64)

filtered_datatest = x_test_images
filtered_datatest[filtered_datatest < 240] = 0
filtered_datatest[filtered_datatest >= 240] = 255
filtered_datatest = filtered_datatest.reshape(-1, 64, 64)

largestsize_train, largestsize_test = preprocess(
    filtered_datatrain, filtered_datatest)
plt.imshow(train_images[4])
plt.imshow(largestsize_train[5])
