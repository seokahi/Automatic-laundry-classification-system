from PIL import Image
from PIL import ImageEnhance
import os


input_folder = '0529/resize/short_sleeve_resize'
output_folder = '0529/contract_image/short_sleeve/'


def adjust_brightness(input_image_path, output_image_path, factor):
    with Image.open(input_image_path) as img:
        # 紐낅룄 議곗젅
        enhancer = ImageEnhance.Brightness(img)
        brightness_img = enhancer.enhance(factor)
        # ��議� 議곗젅
        contrast_enhancer = ImageEnhance.Color(brightness_img)
        output = contrast_enhancer.enhance(factor+5)
        # ����
        output.save(output_image_path)


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 1.5, 0.5, 0.8, 1.2
factor = 1.2
for filename in os.listdir(input_folder):

    if filename.endswith(".jpg") or filename.endswith(".png"):
        print(filename)
        ftitle, fext = os.path.splitext(filename)
        filerename = ftitle + '1.2' + fext
        print(filerename)
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filerename)
        adjust_brightness(input_path, output_path, factor)

print(len(os.listdir(output_folder)))
