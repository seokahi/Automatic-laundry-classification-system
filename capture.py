import serial
import time
import numpy as np
import cv2
import glob
import RPi.GPIO as GPIO 
import os

port_name = glob.glob('/dev/ttyACM*')
print(port_name)
button_state = ''
flag = 0
button_pin = 15
cnt = len(os.listdir('add_data/long_sleeve'))

GPIO.setwarnings(False) 
GPIO.setmode(GPIO.BCM) 

GPIO.setup(button_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)


while True:
    if GPIO.input(button_pin) == GPIO.HIGH:
            ####
            camera = cv2.VideoCapture(0)
            ret, frame = camera.read()
            if camera.isOpened():
                camera.release()
                
            
            filename = "add_data/long_sleeve/long_sleeve{0:04d}.jpg".format(cnt)
            
            cv2.imwrite('result_img/img.jpg',frame)
            imageUrl = 'result_img/img.jpg'
            print('Image captured') 
            frame = [] # frame =0
        
        
            img = cv2.imread(imageUrl)
            ori_img = img
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img[:,:,0] = clahe.apply(img[:,:,0])
            img[:,:,1] = clahe.apply(img[:,:,1])
            img[:,:,2] = clahe.apply(img[:,:,2])
            cv2.imwrite('result_img/CLAHE_res.jpg', img)
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
            cv2.imwrite('result_img/canny_res.jpg', edged)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
            closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
            cv2.imwrite('result_img/mor_res.jpg', closed)

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
            print("x Min: ", x_min)
            print("x Max: ", x_max)
            
            y_min, y_max = 0,0
            value = list()
            for i in range(len(contours_xy)):
                for j in range(len(contours_xy[i])):
                    value.append(contours_xy[i][j][0][1]) 
                    y_min = min(value)
                    y_max = max(value)
            print("y Min: ", y_min)
            print("y Max: ", y_max)


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
            print("automatic resizing: ", rect)

            img_trim = image[y:y+h, x:x+w]
            mask = np.zeros(img.shape[:2], np.uint8)

            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT) 
            
            mask2 = np.where((mask == cv2.GC_FGD) | (
                mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')

            result = cv2.bitwise_and(img, img, mask=mask2)


            result2 = result.copy()

            mask3 = cv2.inRange(result2, np.array([0, 0, 0]), np.array([8, 8, 8]))
            result2[mask3 == 255] = (255, 255, 255)

            result = cv2.resize(result, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)
            result2 = cv2.resize(result2, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite("result.png", result)
            cv2.imwrite(filename, result2)
            print(filename)
            cnt = len(os.listdir('add_data/long_sleeve/'))