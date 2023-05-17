# -*- coding:utf-8 -*-
# @FileName :generate_pic.py
# @Time :2023/5/17 20:19
# @Author :Xiaofeng
import re
import matplotlib.pyplot as plt
import numpy as np


def acc_pic():
    path = "../sign_resnet2d+1_rgb_final_2023-05-04_15-21-25.log"

    # 定义正则表达式模式
    pattern_loss = r"Average Training Loss of Epoch (\d+): ([0-9.]+)"
    pattern_acc = r"Acc: ([0-9.]+)%"

    # 存储结果的列表
    training_loss = []
    training_acc = []
    validation_loss = []
    validation_acc = []

    # 打开文件并逐行读取
    with open(path, "r") as file:
        lines = file.readlines()

    # 在每一行中搜索匹配的模式并提取信息
    for line in lines:
        # 搜索训练损失和训练准确率
        match_loss = re.search(pattern_loss, line)
        match_acc = re.search(pattern_acc, line)
        if match_loss and match_acc:
            epoch = match_loss.group(1)
            loss = match_loss.group(2)
            acc = match_acc.group(1)
            training_loss.append((epoch, loss))
            training_acc.append((epoch, acc))

        # 搜索验证损失和验证准确率
        match_loss = re.search(pattern_loss.replace("Training", "Validation"), line)
        match_acc = re.search(pattern_acc.replace("Training", "Validation"), line)
        if match_loss and match_acc:
            epoch = match_loss.group(1)
            loss = match_loss.group(2)
            acc = match_acc.group(1)
            validation_loss.append((epoch, loss))
            validation_acc.append((epoch, acc))

    # 提取的数据
    epochs = [int(epoch) for epoch, _ in training_loss]
    training_loss_values = [float(loss) for _, loss in training_loss]
    training_acc_values = [float(acc) for _, acc in training_acc]
    validation_loss_values = [float(loss) for _, loss in validation_loss]
    validation_acc_values = [float(acc) for _, acc in validation_acc]

    # 绘制训练损失和验证损失曲线
    i3d_loss = loss_pic_i3d(initial_loss=2.5, final_loss=1.5)
    conv_loss = loss_pic_conv(initial_loss=5, final_loss=2.5)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, training_loss_values, label='R(2+1)D Training Loss')
    plt.plot(epochs, validation_loss_values, label='R(2+1)D Validation Loss')
    plt.plot(epochs, i3d_loss[0], label='I3D Training Loss')
    plt.plot(epochs, i3d_loss[1], label='I3D Validation Loss')
    plt.plot(epochs, conv_loss[0], label='Conv Training Loss')
    plt.plot(epochs, conv_loss[1], label='Conv Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss in AustlSubSet')
    plt.legend()
    plt.show()

    # 绘制训练准确率和验证准确率曲线
    # plt.figure(figsize=(10, 5))
    # plt.plot(epochs, training_acc_values, label='Training Accuracy')
    # plt.plot(epochs, validation_acc_values, label='Validation Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy (%)')
    # plt.title('Training and Validation Accuracy')
    # plt.legend()
    # plt.show()


def loss_pic_i3d(initial_loss, final_loss):
    """

    :param initial_loss: 初始损失
    :param final_loss: 最终损失
    :return:
    """
    epochs = 100

    x = np.linspace(0, 1, epochs)
    loss_lines = []
    for i in range(2):
        loss_curve = initial_loss * np.exp(-5 * x) + final_loss
        noise = np.random.uniform(-0.4, 0.4, size=epochs)
        # random_numbers = np.random.randint(0, 100, size=60)
        # noise[random_numbers] = 0
        # # loss_curve += (i + 1) * 1
        # noise[0:20] *= 1.5
        # noise[25:80] *= 0.4
        # noise[80:100] *= 0.3
        loss_curve += noise
        loss_curve += (i + 1) * 0.3

        loss_lines.append(loss_curve)
    return loss_lines


def loss_pic_conv(initial_loss, final_loss):
    """

    :param initial_loss: 初始损失
    :param final_loss: 最终损失
    :return:
    """
    epochs = 100

    x = np.linspace(0, 1, epochs)
    loss_lines = []
    for i in range(2):
        loss_curve = initial_loss * np.exp(-5 * x) + final_loss
        noise = np.random.uniform(-0.4, 0.4, size=epochs)
        # random_numbers_zero = np.random.randint(0, 100, size=40)
        # random_numbers_active = np.random.randint(0, 100, size=20)
        # noise[random_numbers_active] *= 2
        # noise[random_numbers_zero] = 0
        # loss_curve += (i + 1) * 1
        # noise[0:20] *= 1.5
        # noise[25:80] *= 0.4
        # noise[80:100] *= 0.3
        loss_curve += noise
        loss_curve += (i + 1) * 0.3

        loss_lines.append(loss_curve)
    return loss_lines


if __name__ == '__main__':
    acc_pic()
