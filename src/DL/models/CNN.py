'''
@Author: https://github.com/649453932/
@Date: 2020-04-09 15:59:02
@LastEditTime: 2020-04-09 17:48:16
@LastEditors: Please set LastEditors
@Description: Convolutional Neural Networks for Sentence Classification
@FilePath: /textClassification/src/DL/models/CNN.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchsnooper
from __init__ import *
from src.utils import config


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'CNN'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.save_path = dataset + '/model/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.device = config.device  # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 10000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 50000                                          # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 32                                           # mini-batch大小
        self.pad_size = 400                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.embed = 300                                                # 向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)
        
        self.eps = 1e-8


'''Convolutional Neural Networks for Sentence Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.embed)
#         self.embedding = nn.Embedding(50000, 300)
        self.convs = nn.ModuleList([nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
    
    def forward(self, x):
#         print(x.shape)
        out = self.embedding(x[0])
#         print(out.shape)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
