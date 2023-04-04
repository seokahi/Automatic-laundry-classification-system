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

result2 = result.copy()

mask3 = cv2.inRange(result2, np.array([0,0,0]), np.array([10,10,10]))
result2[mask3 == 255] = (255, 255, 255)

# 결과 출력
cv2.imshow('result', result)
cv2.imshow('result2', result2)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("result.png", result)
cv2.imwrite("result2.png", result2)

img = cv2.imread('result.png')
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

mask = cv2.inRange(image, (1, 1, 1), (255, 255, 255))
image = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("remove_black", image)

# BGR로 된 그랩컷 이미지를 RGB로 바꿈.
# image = cv2.cvtColor(grabcut_image_PIL, cv2.COLOR_BGR2RGB)

img_np = np.asarray(image)
# 이미지 height, width 통합 // kmeans 입력 데이터 조건 맞추기 위해 차원 변경

image = img_np.reshape(img_np.shape[0] * img_np.shape[1], img_np.shape[2])
# 3
img = cv2.imread('result.png')
cv2.imshow("Lenna", img)  # 불러온 이미지를 Lenna라는 이름으로 창 표시

image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
image = image.reshape((image.shape[0] * image.shape[1], 3))  # height, width 통합

mask = np.all(image != [0, 0, 0], axis=-1)  # 검은색(0, 0, 0)이 아닌 픽셀에 True 값 부여
image = image[mask]

# 군집형성의 개수를 3으로 설정
k = 3
# 학습의 동일한 결과를 위해 random_state를 10으로 고정
# k평균 군집화 알고리즘 함수를 활용해 특정 군집 개수만큼 군집화 진행
model = KMeans(n_clusters=k, random_state=10)
# image에 대한 클러스터링 수행
model.fit(image)

# .cluster_centers_ 라는 함수를 통해서 중심점 확인
for center in model.cluster_centers_:
    print(center)

# 중심점 통해비율 구하는 함수


def centroid_histogram(model):
    numLabels = np.arange(0, len(np.unique(model.labels_)) + 1)
    (hist, _) = np.histogram(model.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist


hist = centroid_histogram(model)
print("색상 비율 출력 => ", hist)

# 가장 많은 비율 차지하고 있는 index 추출
max_index = hist.argmax()

# bar 그려주기 위한 함수


def plot_colors(hist, centroids):
    bar = np.zeros((12, 36, 3), dtype="uint8")
    startX = 0
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 36)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 12),
                      color.astype("uint8").tolist(), -1)
        startX = endX
    return bar


bar = plot_colors(hist, model.cluster_centers_)

# 중복값 제거
mylist = bar[0]

# 비율 rgb 담기 위한 my_lists
mylists = []
for a, b, c in mylist:
    if [a, b, c] not in mylists:
        mylists.append([a, b, c])


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



