from keras.models import load_model
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils
import tensorflow as tf

import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import colorsys
from PIL import Image
import tensorflow as tf
from tensorflow import keras

imageUrl = 'images/t-shirt.jpg'
img = cv2.imread(imageUrl)
img = cv2.resize(img, dsize=(0, 0), fx=0.2, fy=0.2,
                 interpolation=cv2.INTER_LINEAR)

mask = np.zeros(img.shape[:2], np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

rect = (53, 139, 538, 583)
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img = img*mask2[:, :, np.newaxis]

tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
b, g, r = cv2.split(img)
rgba = [b, g, r, alpha]
dst = cv2.merge(rgba, 4)

cv2.imwrite("result.png", dst)

grabcut_image = cv2.imread('result.png')

x_pos, y_pos, width, height = cv2.selectROI(
    "location", grabcut_image, False, False)

grabcut_image = grabcut_image[y_pos:y_pos+height, x_pos:x_pos+width]
cv2.imshow("grabcut_image", grabcut_image)
cv2.waitKey(0)
cv2.imwrite("grabcut_image.png", grabcut_image)

