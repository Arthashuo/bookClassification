'''
@Author: xiaoyao jiang
@Date: 2020-04-08 15:35:24
@LastEditTime: 2020-04-09 17:15:35
@LastEditors: Please set LastEditors
@Description: tools
@FilePath: /textClassification/src/utils/tools.py
'''

import logging
import re
import numpy as np
import torch
import time
from datetime import timedelta



def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def create_logger(log_path):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(filename=log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"\s+", "", string)
    # string = re.sub(r"[^\u4e00-\u9fff]", "", string)
    string = re.sub(r"[^\u4e00-\u9fa5^.^,^!^?^:^;^、^a-z^A-Z^0-9]", "", string)
    # string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r",", " , ", string)
    # string = re.sub(r"!", " ! ", string)
    # string = re.sub(r"\(", " \( ", string)
    # string = re.sub(r"\)", " \) ", string)
    # string = re.sub(r"\?", " \? ", string)
    return string.strip()


def strQ2B(ustring):
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return "".join(ss)


def wam(sentence, w2v_model):
    '''
    @description: 通过word average model 生成句向量
    @param {type}
    sentence: 以空格分割的句子
    w2v_model: word2vec模型
    @return:
    '''
    arr = []
    for s in sentence.split():
        if s not in w2v_model.wv.vocab.keys():
            arr.append(np.random.randn(1, 300))
        else:
            arr.append(w2v_model.wv.get_vector(s))
    return np.mean(np.array(arr), axis=0).reshape(300,)


def padding(indice, max_length, pad_idx=0):
    """
    pad 函数
    注意 token type id 右侧pad是添加1而不是0，1表示属于句子B
    """
    pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
    return torch.tensor(pad_indice)