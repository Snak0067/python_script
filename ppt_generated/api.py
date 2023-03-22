# -*- coding:utf-8 -*-
# @FileName :api.py
# @Time :2023/3/21 17:00
# @Author :Xiaofeng
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt, Cm


def loadcsv():
    """
    读取正确词目文件及错误词目csv文件，并转成字典
    :return: id_to_gloss, gloss_to_id
    """
    word_reader = pd.read_csv('rightForm.csv', encoding='utf-8')
    words = np.array(word_reader)
    return words


def generate(words: list):
    # 创建空白演示文稿
    prs = Presentation()

    slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(slide_layout)
    shapes = slide.shapes
    for shape in shapes:
        print(shape.text_frame)
    text_frame = shape.text_frame
    para1 = text_frame.paragraphs[0]
    print(para1)
    # 保存
    prs.save('new_medical_dict.pptx')


if __name__ == '__main__':
    words = loadcsv()
    generate(words)
