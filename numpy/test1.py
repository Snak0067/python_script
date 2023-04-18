# -*- coding:utf-8 -*-
# @FileName :test1.py
# @Time :2023/4/6 11:06
# @Author :Xiaofeng
import numpy as np


def randn_uniform():
    """
    random.uniform(a,b) 返回a,b之间的随机浮点数，若a<=b则范围[a,b]，若a>=b则范围[b,a] ，a和b可以是实数
    random.randint(a,b) 返回a,b之间的整数，范围[a,b]，注意：传入参数必须是整数，a一定要比b小
    """
    for i in range(10):
        random_number = np.random.uniform(-6, -1)
        print(random_number, 10 ** random_number)


def logspace():
    # 对数等比数列
    # np.logspace(start=开始值，stop=结束值，num=元素个数，base=指定对数的底, endpoint=是否包含结束值)
    weight_scales = np.logspace(-4, 0, num=20)
    print(weight_scales)


def np_amax():
    np.random.seed(15)
    a = np.random.randint(1, 10, [2, 3, 3])
    print(a)
    b = np.amax(a, axis=(-1))
    print(b)


def np_add_at_1():
    a = np.array([1, 2, 3, 4])
    b = np.array([1, 2])
    np.add.at(a, [0, 1], b)
    print(a)


def np_add_at_2():
    x = [[0, 4, 1], [3, 2, 4]]
    dW = np.zeros((5, 6))
    np.add.at(dW, x, 1)
    print(dW)


if __name__ == '__main__':
    np_add_at_2()
