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

img = cv2.imread('images/t-shirt.jpg')
mask = np.zeros(img.shape[:2], np.uint8)

# 전경/배경 지정 (알고리즘은 전경과 배경을 기준으로 이미지를 분할합니다)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# 이미지의 중심을 기준으로, 가로 길이와 세로 길이가 이미지 크기의 1/10인 사각형을 초기 ROI로 설정합니다.
rows, cols = img.shape[:2]
rect = (int(cols/10), int(rows/10), int(cols*0.8), int(rows*0.8))

# GrabCut을 실행합니다.
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
# 전경 마스크를 만듭니다. (GrabCut 결과에서 전경을 1, 배경을 0으로 설정합니다.)
fg_mask = np.where((mask == cv2.GC_FGD) | (
    mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')

# 전경 마스크를 모든 채널에 적용합니다.
result = cv2.bitwise_and(img, img, mask=fg_mask)
cv2.imwrite("result.png", result)
plt.imshow(result), plt.colorbar(), plt.show()
