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


if __name__ == '__main__':
    randn_uniform()
