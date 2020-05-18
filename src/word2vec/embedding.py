'''
@Author: xiaoyao jiang
@Date: 2020-04-08 17:22:54
@LastEditTime: 2020-04-09 13:39:46
@LastEditors: Please set LastEditors
@Description: train embedding & tfidf
@FilePath: /textClassification/src/word2vec/embedding.py
'''
import pandas as pd
from gensim import models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

import __init__
from src.utils.config import root_path
from src.utils.tools import create_logger

logger = create_logger(root_path + '/logs/embedding.log')


class SingletonMetaclass(type):
    def __init__(self, *args, **kwargs):
        self.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.__instance is None:
            self.__instance = super(SingletonMetaclass, self).__call__(*args, **kwargs)
            return self.__instance
        else:
            return self.__instance


class Embedding(metaclass=SingletonMetaclass):
    def load_data(self):
        logger.info('load data')
        self.data = pd.concat([
            pd.read_csv(root_path + '/data/train_clean.tsv', sep='\t'),
            pd.read_csv(root_path + '/data/dev_clean.tsv', sep='\t'),
            pd.read_csv(root_path + '/data/test_clean.tsv', sep='\t')
        ])
        self.stopWords = open(root_path + '/data/stopwords.txt').readlines()

    def trainer(self):
        logger.info('train tfidf')
        count_vect = TfidfVectorizer(stop_words=self.stopWords, max_df=0.6, ngram_range=(1, 2))
        self.tfidf = count_vect.fit(self.data.text)
        logger.info('train word2vec')
        self.w2v = models.Word2Vec(min_count=2,
                                   window=2,
                                   size=300,
                                   sample=6e-5,
                                   alpha=0.03,
                                   min_alpha=0.0007,
                                   negative=15,
                                   workers=4,
                                   iter=7)
        self.w2v.build_vocab(self.data)
        self.w2v.train(self.data,
                       total_examples=self.w2v.corpus_count,
                       epochs=15,
                       report_delay=1)
        logger.info('train fast')
        self.fast = models.FastText(self.data,
                                    size=300,
                                    window=3,
                                    min_count=1,
                                    iter=10,
                                    min_n=3,
                                    max_n=6,
                                    word_ngrams=2)

    def saver(self):
        logger.info('save tfidf model')
        joblib.dump(self.tfidf, root_path + '/model/embedding/tfidf')
        logger.info('save w2v model')
        self.w2v.save(root_path + '/model/embedding/w2v')
        logger.info('save fast model')
        self.fast.save(root_path + '/model/embedding/fast')

    def load(self):
        logger.info('load tfidf model')
        self.tfidf = joblib.load(root_path + '/model/embedding/tfidf')
        logger.info('load w2v model')
        self.w2v = models.KeyedVectors.load(root_path + '/model/embedding/w2v')
        logger.info('load fast model')
        self.fast = models.FastText.load(root_path + '/model/embedding/fast')


if __name__ == "__main__":
    em = Embedding()
    em.load_data()
    em.trainer()
    em.saver()
