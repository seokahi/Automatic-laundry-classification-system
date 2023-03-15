
# kmeans
from PIL import Image
from keras.models import load_model
from keras.datasets import mnist
import numpy
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils
import tensorflow as tf

import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import colorsys
import os
img = cv2.imread('./result.png')
cv2.imshow("Lenna", img)  # 불러온 이미지를 Lenna라는 이름으로 창 표시.

print(img.shape)
img = img.reshape(50, 50, 3)
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print("이미지 rgb", image)
image = image.reshape((image.shape[0] * image.shape[1], 3))  # height, width 통합
print(image.shape)


k = 3  # 예제는 5개로 나누겠습니다
# 결과 항상 고정
clt = KMeans(n_clusters=k, random_state=10)
print(clt)
clt.fit(image)

# .cluster_centers_ 라는 함수를 통해서 좌표값을 확인할 수 있다.
for center in clt.cluster_centers_:
    print(center)


def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist


hist = centroid_histogram(clt)

print(hist.shape)
max_index = hist.argmax()


def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((12, 36, 3), dtype="uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 36)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 12),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar


bar = plot_colors(hist, clt.cluster_centers_)
# 중복값 제거

print("타입", bar.shape)
print("bar0", type(bar[1]))
print("타입", bar[1][1])
mylist = bar[0]
print(bar[0])
print(bar[1])

mylists = []
for a, b, c in mylist:
    if [a, b, c] not in mylists:
        mylists.append([a, b, c])

print(mylists[1])
print(mylists[2])
# 옷의 밝기 인식을 위한 명도 구하기
h1, s1, v1 = colorsys.rgb_to_hsv(
    mylists[max_index][0]/255, mylists[max_index][1]/255, mylists[max_index][2]/255)

print("명도", v1)
# 밝음
if (v1 > 0.5):
    flag = 1
# 어두움
else:
    flag = 0

print("밝은 옷: 1 어두운 옷: 0 =>", flag)
# show our color bartS
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()
