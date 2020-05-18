'''
@Author: https://github.com/649453932/
@LastEditTime: 2020-04-09 17:43:56
@LastEditors: Please set LastEditors
@Description: Recurrent Convolutional Neural Networks for Text Classification
@FilePath: /textClassification/src/DL/models/RCNN.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'TextRCNN'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = dataset + '/model/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 1.0                                              # 随机失活
        self.require_improvement = 10000                                # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 50000                                            # 词表大小，在运行时赋值
        self.num_epochs = 10                                            # epoch数
        self.embed = 300                                                # 向量维度
        self.batch_size = 32                                            # mini-batch大小
        self.pad_size = 400                                             # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.hidden_size = 512                                          # lstm隐藏层
        self.num_layers = 1                                             # lstm层数
        self.eps = 1e-8


'''Recurrent Convolutional Neural Networks for Text Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=0)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.hidden_size * 2 + config.embed, config.num_classes)

    def forward(self, x):
        embed = self.embedding(x[0])  # [batch_size, seq_len, embeding]=[64, 32, 64]
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out) 
        out = out.squeeze()
        out = self.fc(out)
        return out