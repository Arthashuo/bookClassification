'''
@Author: your name
@Date: 2020-04-08 17:21:28
@LastEditTime: 2020-04-22 20:29:42
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /textClassification/src/data/dataset.py
'''

import numpy as np
import pandas as pd
import json
from __init__ import *
from src.utils import config
from src.utils.tools import create_logger, wam
from src.word2vec.embedding import Embedding
logger = create_logger(config.log_dir + 'data.log')


class MLData(object):
    def __init__(self):
        self.em = Embedding()
        self.em.load()

    def process_data(self, method='word2vec'):
        logger.info('load data')
        train = pd.read_csv(config.root_path + '/data/train_clean.tsv', sep='\t')
        dev = pd.read_csv(config.root_path + '/data/dev_clean.tsv', sep='\t')

        train['category_id'] = train['label'].factorize()[0]
        dev['category_id'] = dev['label'].factorize()[0]
        self.category_id_df = train[['label', 'category_id']].drop_duplicates()
        with open(config.root_path + '/data/label2id.json', 'w') as f:
            json.dump({k: v for k, v in zip(self.category_id_df['label'], self.category_id_df['category_id'])}, f)
        X_train = self.get_feature(train, method)
        X_test = self.get_feature(dev, method)
        y_train = train['category_id'].values
        y_test = dev['category_id'].values
        return X_train, X_test, y_train, y_test

    def get_feature(self, data, method='word2vec'):
        if method == 'tfidf':
            return self.em.tfidf.transform(data['text'])
        elif method == 'word2vec':
            # return [np.array(wam(x, self.em.w2v)) for x in data['text'].values.tolist()]
            return data['text'].apply(lambda x: wam(x, self.em.w2v)[0])
        elif method == 'fasttext':
            return data['text'].apply(lambda x: wam(x, self.em.fast)[0])
        else:
            NotImplementedError
