# -*- coding:utf-8 -*-
# @FileName :make_all_pic.py
# @Time :2023/5/19 23:17
# @Author :Xiaofeng
from PIL import Image
import os


def merge_images(image_paths, output_path, rows, cols):
    images = []
    for path in image_paths:
        image = Image.open(path)
        images.append(image)

    width, height = images[0].size
    merged_width = width * cols
    merged_height = height * rows

    merged_image = Image.new("RGB", (merged_width, merged_height))

    for i, image in enumerate(images):
        row = i // cols
        col = i % cols
        x = col * width
        y = row * height
        merged_image.paste(image, (x, y))

    merged_image.save(output_path)
    print("总览图片生成成功！")


# 示例使用
image_folder = "initial_video_frames"
# image_folder = "video_frames"
output_path = "merged_image.jpg"
rows = 3
cols = 8

# 获取文件夹中所有图片的路径
image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if
               filename.endswith(".jpg")]

# 调用合并图片函数
merge_images(image_paths, output_path, rows, cols)
