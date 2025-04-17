"""
generate plate numbers
"""

import numpy as np
import cv2, os
from glob import glob
from typing import List
# 省份
provinces = ["京", "津", "冀", "晋", "蒙", "辽", "吉", "黑", "沪",
             "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘",
             "粤", "桂", "琼", "渝", "川", "贵", "云", "藏", "陕",
             "甘", "青", "宁", "新"]

# "港", "澳", "使", "领", "学", "警", "挂"]
digits = ['{}'.format(x + 1) for x in range(9)] + ['0']

# 英文，没有I、O两个字符
letters = [chr(x + ord('A')) for x in range(26) if not chr(x + ord('A')) in ['I', 'O']]
# print('letters', digits + letters)

# 随机选取
def random_select(data):
    return data[np.random.randint(len(data))]

# 蓝牌
def generate_plate_number_blue(length=7):
    plate = random_select(provinces)         # 第1位：省份
    plate += random_select(letters)          # 第2位：发牌机关代码（字母）

    # 第3-7位：车牌序号（按三种规则之一随机）
    rule_type = np.random.choice([1, 2, 3])  # 规则 a/b/c

    if rule_type == 1:
        # a) 全数字
        serial = ''.join(random_select(digits) for _ in range(length-2))
    elif rule_type == 2:
        # b) 1个字母，其余是数字
        letter_pos = np.random.randint(length-2)
        serial = ''
        for i in range(length-2):
            if i == letter_pos:
                serial += random_select(letters)
            else:
                serial += random_select(digits)
    else:
        # c) 2个字母，其余是数字
        letter_pos = sorted(np.random.choice(range(length-2), 2, replace=False))
        serial = ''
        for i in range(length-2):
            if i in letter_pos:
                serial += random_select(letters)
            else:
                serial += random_select(digits)

    plate += serial
    return plate

def generate_plate_number(st: List[str]) -> str:
    """
    生成黄牌号牌，支持自定义前缀和后缀。
    通常用于挂车、工程车、教练车等。

    :param length: 总长度（默认为7位号牌）
    :param st: 前缀字符串
    :param end: 后缀字符串
    :return: 拼接后的黄牌号
    """
    base_plate = generate_plate_number_blue(length=len(st))
    plate_chars = list(base_plate)
    for i, ch in enumerate(st):
        if ch != '*':
            plate_chars[i] = ch
    return ''.join(plate_chars)

def board_bbox(polys):
    x1, y1 = np.min(polys, axis=0)
    x2, y2 = np.max(polys, axis=0)

    return [x1, y1, x2, y2]

if __name__ == '__main__':
    pass