import cv2
import os

# 이미지가 저장된 폴더 경로
origin_image_path = '0529/dataset/short_sleeve'
save_path = '0529/short_sleeve_resize/'

# 이미지 크기 조정 함수


def resize_img(folder_path, save_folder_path, size):
    cnt = 0
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, size)
            save_folder = save_folder_path + \
                'short_sleeve_{0:04d}'.format(cnt) + '.jpg'
            cv2.imwrite(save_folder, img)
            cnt += 1


# 모든 이미지를 256x256 크기로 조정
resize_img(origin_image_path, save_path, (256, 256))
print(len(os.listdir(save_path)))
