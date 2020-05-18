'''
@Author: xiaoyao jiang
@Date: 2020-04-08 15:17:27
@LastEditTime: 2020-04-08 19:45:06
@LastEditors: Please set LastEditors
@Description: all model config
@FilePath: /textClassification/src/utils/config.py
'''


import torch
import os

# generate config
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(os.path.split(curPath)[0])[0]
# root_path = '/home/user10000281/notespace/textClassification/'
train_file = root_path + '/data/train_clean.tsv'
dev_file = root_path + '/data/dev_clean.tsv'
test_file = root_path + '/data/test_clean.tsv'
stopWords_file = root_path + '/data/stopwords.txt'
log_dir = root_path + '/logs/'

# generate dl config
is_cuda = True
max_length = 400
device = torch.device('cuda') if is_cuda else torch.device('cpu')
