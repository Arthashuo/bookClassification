# coding: UTF-8
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification
from __init__ import *
from src.utils import config


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'xlnet'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + 'model/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.device = config.device  # 设备  
        
        self.require_improvement = 10000                                # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数

        self.num_epochs = 30                                          # epoch数
        self.batch_size = 32                                           # mini-batch大小
        self.pad_size = 400                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                      # 学习率
        self.bert_path = dataset + 'model/roberta/'
        self.tokenizer = RobertaTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.eps = 1e-8
        self.gradient_accumulation_steps = 1


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        model_config = RobertaConfig.from_pretrained(config.bert_path, num_labels=config.num_classes)
        self.roberta = RobertaForSequenceClassification.from_pretrained(config.bert_path, config=model_config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[1]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        token_type_ids = x[2]
        _, pooled = self.roberta(context, attention_mask=mask, token_type_ids=token_type_ids)
        out = self.fc(pooled)
        return out