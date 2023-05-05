import serial
import time
import tensorflow as tf
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import colorsys
from tensorflow import keras
import tensorflow.keras.preprocessing.image as tpi
import glob

port_name = glob.glob('/dev/ttyACM*')
print(port_name)
ser = serial.Serial(port_name[0], 9600, timeout=1) # 아두이노와 시리얼 통신 ser설정
button_state = ''
flag = 0

while True:
    start = time.time()
    button_state = ''
    if ser.in_waiting > 0 and flag == 0:
        button_state = ser.readline().decode('ascii').rstrip() 
        print(button_state)
        if button_state is not None:
            flag = 1
        print(type(button_state))
    print("serial connection time: ", time.time() - start)
    if button_state == '1': # 버튼이 눌리면
        ####
        start = time.time()
        camera = cv2.VideoCapture(0)
        ret, frame = camera.read()
        if camera.isOpened():
            camera.release()
            
        cv2.imwrite('result_image/image.jpg',frame)
        print('Image captured') # 이미지 캡처 완료 메시지 출력
        ser.write(b"done\n")
        frame = [] # frame =0
        imageUrl = 'result_image/image.jpg'
        print("image capture and save part: ", start - time.time())
        ####
        start = time.time()
        img = cv2.imread(imageUrl)
        alpha2 = 50
        array = np.full(img.shape, (alpha2, alpha2, alpha2), dtype=np.uint8)
        img = cv2.add(img, array)
        # img = np.clip(img + (img - 128) * alpha2, 0, 255).astype(np.uint8)
        print(len(img))
        cv2.imwrite('result_image/image_daebi.jpg',img)
        print("image read and contrast processing part: ", start - time.time())
        ####
        start = time.time()
        # img = cv2.resize(img, dsize=(0, 0), fx=0.2, fy=0.2,
        #                 interpolation=cv2.INTER_LINEAR)
        # rect = cv2.selectROI("location", img, False, False)
        # print(rect)
        rect = (70, 60, 480, 390) #ROIl
        # rect = (70,20,550,400)
        cv2.destroyAllWindows()

        mask = np.zeros(img.shape[:2], np.uint8)

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        print("ROI selection and grabCut processing part: ", time.time() - start)
        ####
        start = time.time()
        # cv2.grabCut(img, mask, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

        mask2 = np.where((mask == cv2.GC_FGD) | (
            mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')

        result = cv2.bitwise_and(img, img, mask=mask2)


        result2 = result.copy()

        mask3 = cv2.inRange(result2, np.array([0, 0, 0]), np.array([8, 8, 8]))
        result2[mask3 == 255] = (255, 255, 255)
        # 결과 출력
        # cv2.imshow('result', result)
        # cv2.imshow('result2', result2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite("result.png", result)
        cv2.imwrite("result2.png", result2)
        print("mask processing part: ", start - time.time())
        ####
        start = time.time()
        #####################################################################################

        ########################################################################################################

        ####################################################################################### 그랩 ? ?  고리 ? ################################################################################

        ####################################################################################### k-means ?  고리 ? ################################################################################

        img = cv2.imread('result2.png')
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # checkpoint##########################
        mask = cv2.inRange(image, (0, 0, 0), (255, 255, 255))
        image = cv2.bitwise_and(image, image, mask=mask)

        # image = cv2.cvtColor(grabcut_image_PIL, cv2.COLOR_BGR2RGB)

        img_np = np.asarray(image)
  
        image = img_np.reshape(img_np.shape[0] * img_np.shape[1], img_np.shape[2])
        image = image.reshape((image.shape[0] * image.shape[1], 3))  # height, width ?  ?  

        mask = np.all(image != [255, 255, 255], axis=-1)
        image = image[mask]

        k = 3
        model = KMeans(n_clusters=k, random_state=10)
        model.fit(image)

        for center in model.cluster_centers_:
            print(center)

  

        def centroid_histogram(model):
            numLabels = np.arange(0, len(np.unique(model.labels_)) + 1)
            (hist, _) = np.histogram(model.labels_, bins=numLabels)

            hist = hist.astype("float")
            hist /= hist.sum()

            return hist


        hist = centroid_histogram(model)
        print("?  ?   비율 출력 => ", hist)

        #  ??   많 ?? 비율 차 ???   ? ?  ?   index 추출
        max_index = hist.argmax()

        # bar 그려주기 ?  ?   ?  ?  


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

        #------------------------------------------------------------
        bar = plot_colors(hist, model.cluster_centers_)

        # 중복 ? ?   ?
        mylist = bar[0]

        # 비율 rgb ?   ? ?  ?   my_lists
        mylists = []
        for a, b, c in mylist:
            if [a, b, c] not in mylists:
                mylists.append([a, b, c])


        # ?  ?   밝기 ?  ?  ?   ?  ?   명도 구하 ?
        h1, s1, v1 = colorsys.rgb_to_hsv(
            mylists[max_index][0]/255, mylists[max_index][1]/255, mylists[max_index][2]/255)

        print("명도", v1)
        # 밝음
        if (v1 > 0.5):
            # flag = 1
            flag = "light"
        # ?  ?  ???
        else:
            # flag = 0
            flag= "dark"

        print("%s" %flag)
        # show our color bartS
        # plt.figure()
        # plt.axis("off")
        # plt.imshow(bar)
        # plt.show()

        ####################################################################################### k-means ?  고리 ? ################################################################################


        ####################################################################################### fashion-mnist ?  고리 ? ################################################################################
        MNIST_model = tf.keras.models.load_model('/home/pi/caps/custom_v6.h5', compile=False)
        img = tf.keras.preprocessing.image.load_img(
            './result2.png',
            target_size=(28, 28),
            color_mode='grayscale'
        )
        # img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        # img_array = tf.keras.preprocessing.img_to_array(img)

        image = np.expand_dims(img_array, axis=-1)
        image = np.expand_dims(img_array, axis=0)

        predictions = MNIST_model.predict(image)
        score = tf.nn.softmax(predictions[0])

        class_index = np.argmax(predictions[0])
        class_names = ['bra', 'pantie', 'pants', 'shirt',
                    'short pants', 'skirt', 'socks', 't-shirt']

        print(class_index)
        print('Predicted class:', class_names[class_index])

        # plt.imshow(img)
        # plt.show()

        if (class_names[class_index] == "bra" or class_names[class_index] == "pantie"):
            print("under")
            ser.write(b'1\n')
        elif ((class_names[class_index] != "bra" or class_names[class_index] == "pantie") and flag == "dark"):
            print("dark clothes")
            ser.write(b'2n')
        else:
            print("light clothes")
            ser.write(b'3\n')
        
        button_state = ''
        flag = 0
        ####################################################################################### fasjion-mnist ?  고리 ? ################################################################################
        