from PIL import Image
from PIL import ImageEnhance
import os

input_folder = '/content/drive/MyDrive/cap/week13/short_sleeve'  # 입력 폴더 경로
# input_folder = '/content/drive/MyDrive/cap/short_sleeve'
output_folder = '/content/drive/MyDrive/cap/week13/short_sleeve_out_daebi/'  # 출력 폴더 경로


def adjust_brightness(input_image_path, output_image_path, factor):
    with Image.open(input_image_path) as img:
        # 명도 조절
        enhancer = ImageEnhance.Brightness(img)
        brightness_img = enhancer.enhance(factor)
        # 대조 조절
        contrast_enhancer = ImageEnhance.Color(brightness_img)
        output = contrast_enhancer.enhance(factor+5)
        # 저장
        output.save(output_image_path)

# 폴더 내의 모든 이미지에 대해 명도를 조절
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

factor = 0.5
for filename in os.listdir(input_folder):
    # 이미지 파일인 경우에만 처리
    if filename.endswith(".jpg") or filename.endswith(".png"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        adjust_brightness(input_path, output_path, factor)
print(len(os.listdir(output_folder)))

# # 이미지 로드
# img = Image.open("image.jpg")

# # 명도를 1.5배로 증가시키기
# enhancer = ImageEnhance.Brightness(img)
# img = enhancer.enhance(1.5)

# # 이미지 저장
# img.save("bright_image.jpg")