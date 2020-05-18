'''
@Author: xiaoyao jiang
@Date: 2020-04-08 15:16:46
@LastEditTime: 2020-04-08 17:20:58
@LastEditors: Please set LastEditors
@Description: clean data
@FilePath: /textClassification/src/dataProcess/clean.py
'''
import jieba
import pandas as pd
from tqdm import tqdm

from __init__ import *
from src.utils import config
from src.utils.tools import clean_str, create_logger, strQ2B

tqdm.pandas()

logger = create_logger(config.log_dir + 'clean.log')


class CLEAN(object):
    def load_data(self, file_path):
        return pd.read_csv(file_path, sep='\t').dropna().reset_index(drop=True)

    def preprocessor(self, data: pd.DataFrame):
        data = data[data['desc'].str.len() > 1]
        data['text'] = data['title'] + data['desc']
        data['text'] = data['text'].progress_apply(
            lambda x: " ".join(jieba.cut(clean_str(strQ2B(x)))))
        return data[['text', 'label']]

    def save(self, data, save_name):
        data.to_csv('./data/{}.tsv'.format(save_name), sep='\t', index=False)

    def clean_together(self):
        logger.info('reading data...')
        train = cl.load_data(config.train_file)
        dev = self.load_data(config.dev_file)
        test = self.load_data(config.test_file)
        logger.info('process data...')
        train = self.preprocessor(train)
        dev = self.preprocessor(dev)
        test = self.preprocessor(test)
        logger.info('save data...')
        self.save(train, 'train_clean')
        self.save(dev, 'dev_clean')
        self.save(test, 'test_clean')


if __name__ == "__main__":
    cl = CLEAN()
    cl.clean_together()
