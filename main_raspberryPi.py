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
import RPi.GPIO as GPIO
port_name = glob.glob('/dev/ttyACM*')
print(port_name)
ser = serial.Serial(port_name[0], 9600, timeout=1)
button_pin = 15

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)


GPIO.setup(button_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

time.sleep(1)
while True:
    time.sleep(0.1)
    
    if GPIO.input(button_pin) == GPIO.HIGH: 

        camera = cv2.VideoCapture(0)
        ret, frame = camera.read()
        if camera.isOpened():
            camera.release()
            
        cv2.imwrite('result_image/image.jpg',frame)
        imageUrl = 'result_image/image.jpg'
        print('Image captured') 

        frame = [] 
        img = cv2.imread(imageUrl)
        ori_img = img
        start = time.time()
        img = cv2.imread(imageUrl)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img[:,:,0] = clahe.apply(img[:,:,0])
        img[:,:,1] = clahe.apply(img[:,:,1])
        img[:,:,2] = clahe.apply(img[:,:,2])

        img = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        ori_img = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        print("Origin image size: ", img.shape)
        image_gray = cv2.imread('result_img/img.jpg', cv2.IMREAD_GRAYSCALE)
        image_gray = cv2.resize(image_gray, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite('result_img/gray.jpg', image_gray)
        image_gray = cv2.equalizeHist(image_gray)
        cv2.imwrite('result_img/gray_nor.jpg', image_gray)
        image = img[:290, 0:545]
        image_gray = image_gray[:290, 0:545]
        blur = cv2.GaussianBlur(image_gray, ksize=(3,3), sigmaX=0)
        ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

        edged = cv2.Canny(blur, 10, 250)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total = 0

        contours_xy = np.array(contours)
        contours_xy.shape

        x_min, x_max = 0,0
        value = list()
        for i in range(len(contours_xy)):
            for j in range(len(contours_xy[i])):
                value.append(contours_xy[i][j][0][0])
                x_min = min(value)
                x_max = max(value)

        y_min, y_max = 0,0
        value = list()
        for i in range(len(contours_xy)):
            for j in range(len(contours_xy[i])):
                value.append(contours_xy[i][j][0][1]) 
                y_min = min(value)
                y_max = max(value)

        
        x = x_min
        y = y_min
        w = x_max-x_min
        h = y_max-y_min

        if x < 10:
            x = 0
        else:
            x = x - 10
        if y < 10:
            y = 10
        else:
            y = y - 10
        if w > 630:
            w = 640
        else:
            w = w + 10
        if h > 470:
            h = 480
        else:
            h = h + 10
        rect = (x, y, w, h)

        img_trim = image[y:y+h, x:x+w]
        cv2.imwrite('result_img/autoROI_res.jpg', img_trim)

        mask = np.zeros(img.shape[:2], np.uint8)

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        print("ROI selection and grabCut processing part: ", time.time() - start)
        start = time.time()

        mask2 = np.where((mask == cv2.GC_FGD) | (
            mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')

        result = cv2.bitwise_and(img, img, mask=mask2)


        result2 = result.copy()

        mask3 = cv2.inRange(result2, np.array([0, 0, 0]), np.array([8, 8, 8]))
        result2[mask3 == 255] = (255, 255, 255)
        
        ## k-means
        image = cv2.imread('result2.png')

        k = 3
        image = image.reshape((image.shape[0] * image.shape[1], 3)) 

        mask = np.all(image != [255, 255, 255], axis=-1)
        image = image[mask]
        model = KMeans(n_clusters=k, random_state=10)

        model.fit(image)

  

        def centroid_histogram(model):
            numLabels = np.arange(0, len(np.unique(model.labels_)) + 1)
            (hist, _) = np.histogram(model.labels_, bins=numLabels)

            hist = hist.astype("float")
            hist /= hist.sum()

            return hist


        hist = centroid_histogram(model)

        max_index = hist.argmax()



        def plot_colors(hist, centroids):
            bar = np.zeros((12, 36, 3), dtype="uint8")
            startX = 0
            for (percent, color) in zip(hist, centroids):
                endX = startX + (percent * 36)
                cv2.rectangle(bar, (int(startX), 0), (int(endX), 12),
                            color.astype("uint8").tolist(), -1)
                startX = endX
            return bar

        bar = plot_colors(hist, model.cluster_centers_)

        mylist = bar[0]

        mylists = []
        for a, b, c in mylist:
            if [a, b, c] not in mylists:
                mylists.append([a, b, c])

        h1, s1, v1 = colorsys.rgb_to_hsv(
            mylists[max_index][0]/255, mylists[max_index][1]/255, mylists[max_index][2]/255)

        if (v1 > 0.44):
            flag = "light"
        else:
            flag= "dark"

        MNIST_model = tf.keras.models.load_model('/custom_final.h5', compile=False)
        img = tf.keras.preprocessing.image.load_img(
            './result2.png',
            target_size=(28, 28),
            color_mode='grayscale'
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)

        image = np.expand_dims(img_array, axis=-1)
        image = np.expand_dims(img_array, axis=0)

        predictions = MNIST_model.predict(image)
        score = tf.nn.softmax(predictions[0])

        class_index = np.argmax(predictions[0])
        class_names = ['long_pants', 'long_sleeve', 'short_pants', 'short_sleeve']

        print(class_index)
        print('Predicted class:', class_names[class_index])

   
        if (class_names[class_index] == "bra" or class_names[class_index] == "pantie"):
            print("under")
            ser.write(b'1\n')
        elif ((class_names[class_index] != "bra" or class_names[class_index] == "pantie") and flag == "dark"):
            print("dark clothes")
            ser.write(b'2n')
        else:
            print("light clothes")
            ser.write(b'3\n')
        
        GPIO.input(button_pin) == GPIO.LOW
   