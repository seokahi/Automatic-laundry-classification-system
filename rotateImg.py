from PIL import Image
import os

input_folder = 'C:/Python/OpenCV/AutoCrawler-master/download/socks'  # 입력 폴더 경로
output_folder = 'C:/Python/OpenCV/AutoCrawler-master/download/socks-ouput2'  # 출력 폴더 경로


def rotate_images(input_folder, ouput_folder, degrees):
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(ouput_folder, filename)

            with Image.open(input_path) as img:
                output = img.rotate(degrees)
                output.save(output_path)

# 폴더 내의 모든 이미지 회전시키기
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

degrees = 90
rotate_images(input_folder, output_folder, degrees)
degrees = 180
rotate_images(input_folder, output_folder, degrees)
degrees = 270
rotate_images(input_folder, output_folder, degrees)
print(len(os.listdir(output_folder)))