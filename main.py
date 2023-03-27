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

rect = cv2.selectROI("location", img, False, False)


cv2.destroyAllWindows()

# GrabCut 실행을 위한 초기 마스크 생성
mask = np.zeros(img.shape[:2], np.uint8)

# ROI 영역과 그 외 영역을 지정
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# GrabCut 알고리즘 실행
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# 결과로부터 전경 마스크 생성
mask2 = np.where((mask == cv2.GC_FGD) | (
    mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')

# 전경 마스크를 이미지에 적용하여 전경만 추출
result = cv2.bitwise_and(img, img, mask=mask2)

# 결과 출력
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("result.png", result)


