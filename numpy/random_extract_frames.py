import cv2
import os
import random


def extract_frames(video_path, output_dir, num_frames=24):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 确保视频文件成功打开
    if not cap.isOpened():
        print("无法打开视频文件：", video_path)
        return

    # 确定视频帧的总数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 随机选择要提取的帧的索引
    frame_indices = random.sample(range(0, total_frames), num_frames)

    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 逐帧提取和裁剪
    for i, frame_index in enumerate(frame_indices):
        # 设置视频帧的读取位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # 读取视频帧
        ret, frame = cap.read()

        # 确保成功读取视频帧
        if not ret:
            print("无法读取视频帧：", frame_index)
            continue

        # 生成保存的文件名
        output_filename = os.path.join(output_dir, f"{os.path.basename(video_path)}_{i}.jpg")

        # 保存裁剪后的帧
        cv2.imwrite(output_filename, frame)

    # 关闭视频文件
    cap.release()

    print("视频帧提取和裁剪完成！")


def random_crop_extract_frames(video_path, output_dir, crop_size=(224, 224), num_frames=16):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 确保视频文件成功打开
    if not cap.isOpened():
        print("无法打开视频文件：", video_path)
        return

    # 确定视频帧的总数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 随机选择要提取的帧的索引
    frame_indices = random.sample(range(0, total_frames), num_frames)

    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 逐帧提取和裁剪
    for i, frame_index in enumerate(frame_indices):
        # 设置视频帧的读取位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # 读取视频帧
        ret, frame = cap.read()

        # 确保成功读取视频帧
        if not ret:
            print("无法读取视频帧：", frame_index)
            continue

        # 随机裁剪帧
        frame_height, frame_width, _ = frame.shape
        x = random.randint(0, frame_width - crop_size[0])
        y = random.randint(0, frame_height - crop_size[1])
        cropped_frame = frame[y:y + crop_size[1], x:x + crop_size[0]]

        # 生成保存的文件名
        output_filename = os.path.join(output_dir, f"{os.path.basename(video_path)}_{i}.jpg")

        # 保存裁剪后的帧
        cv2.imwrite(output_filename, cropped_frame)

    # 关闭视频文件
    cap.release()

    print("视频帧提取和裁剪完成！")


# 调用示例
video_path = ["signer20_sample114_color.mp4", "signer4_sample323_color.mp4", "signer7_sample1625_color.mp4"]
output_dir = "video_frames"

for path in video_path:
    # extract_frames(path, "initial_video_frames", 8)
    random_crop_extract_frames(path, output_dir, crop_size=(280, 280), num_frames=8)
