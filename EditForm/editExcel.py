# -*- coding:utf-8 -*-
# @FileName :editExcel.py
# @Time :2023/5/12 9:50
# @Author :Xiaofeng
import subprocess

import cv2


def get_video_png(video_path, png_path, zhen_num=1):
    """
    获取视频封面
    :param video_path: 视频文件路径
    :param png_path: 截取图片存储路径
    :param zhen_num: 指定截取视频第几帧
    :return:
    """
    vidcap = cv2.VideoCapture(video_path)
    # 获取帧数
    zhen_count = vidcap.get(7)

    if zhen_num > zhen_count:
        zhen_num = 1
    print(f"zhen_count = {zhen_count} | last zhen_num = {zhen_num}")

    # 指定帧
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, zhen_num)

    success, image = vidcap.read()
    imag = cv2.imwrite(png_path, image)


if __name__ == "__main__":
    zhen_num = 1
    video_path = 'D:/Code/PythonCode/Chengfeng_backend_system/Chengfeng_backend_system/data-prepare/data' \
                 '/raw_data/signer0_sample12_color.mp4'
    png_path = 'D:/Code/PythonCode/Chengfeng_backend_system/Chengfeng_backend_system/data-prepare/data' \
               f'/picture_cover/signer0_sample12_color_{zhen_num}.png'
    get_video_png(video_path, png_path, zhen_num)
