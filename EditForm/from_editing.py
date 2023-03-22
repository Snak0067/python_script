import numpy as np
import pandas as pd


def loadcsv():
    """
    读取正确词目文件及错误词目csv文件，并转成字典
    :return: id_to_gloss, gloss_to_id
    """
    wrong_reader = pd.read_csv('wrongForm.csv', encoding='utf-8')
    right_reader = pd.read_csv('rightForm.csv', encoding='utf-8')
    id_to_gloss = dict(zip(wrong_reader['index'], wrong_reader['gloss']))
    gloss_to_id = dict(zip(right_reader['gloss'], right_reader['index']))
    return id_to_gloss, gloss_to_id


def updateText(filename):
    """
    依据词目字典,将需替换glossId的文件进行id替换
    :param filename: 需替换的文件的路径
    :return: 缺失的 gloss词目
    """
    id_to_gloss, gloss_to_id = loadcsv()
    # print(20006,id_to_gloss[20006],gloss_to_id[id_to_gloss[20006])

    new_article = []
    with open(filename, 'r') as f:
        article = f.readlines()
    flag_touched = False

    lose_word = []
    for words in article:
        words = words.strip('\n')
        # 匹配到Takes则之后进行glossId的替换,flag->True
        if "category name" in words and "Takes" in words:
            flag_touched = True
        if flag_touched and "class name=" in words:
            first_quotes = words.find('"')
            last_quotes = words.rfind('"')
            # 在txt内容中截取gloss_id
            gloss_id = int(words[first_quotes + 1:last_quotes])
            gloss = id_to_gloss[gloss_id]
            # print(gloss_id, id_to_gloss[gloss_id], gloss_to_id[id_to_gloss[gloss_id]])

            if gloss in gloss_to_id:
                new_words = words[:first_quotes] + str(gloss_to_id[gloss]) + words[last_quotes:]
                new_article.append(new_words)
            else:
                lose_word.append(id_to_gloss[gloss_id])
        else:
            new_article.append(words)
    f.close()

    # 将替换后的文本写入新的文本文件
    with open(filename + '_updated', 'w') as file:
        [file.write(str(item) + '\n') for item in new_article]

    print(lose_word)


if __name__ == '__main__':
    updateText('form/0303cimu.cmpro')
    updateText('form/0303cimu_bak.cmpro')
