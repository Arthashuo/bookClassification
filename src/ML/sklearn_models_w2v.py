import numpy as np
import pandas as pd
import jieba
import requests
import os
from collections import Counter
import lightgbm as lgb
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from gensim.models import Word2Vec
from sklearn.svm import SVC
from sklearn import metrics
from tqdm import tqdm
from __init__ import *
from src.utils import config
from src.utils.tools import create_logger, wam
from src.Embedding.embedding import Embedding
import pickle
import warnings
from sklearn.externals import joblib
from gensim.models import KeyedVectors

from __init__ import *
from src.utils import config
from src.utils.tools import create_logger, Translate_array
from src.Pre_process.mlData import MLData
from src.utils.config import root_path
from gensim import models
logger = create_logger(
    config.log_dir + 'sklearn_models_W2V_embedding_50000.log')

# em = Embedding()  # 创建embedding类的对象
# em.load()
w2v = models.KeyedVectors.load(
    root_path + '/src/Embedding/models/w2v_model')

print("输出词表的个数{}".format(len(w2v.wv.vocab.keys())))
embedding_flag = "word2vec_50000"
train = pd.read_csv(root_path +
                    '/data/train_clean.tsv', sep='\t')
# dev = pd.read_csv(root_path + '/data/dev_clean.tsv', sep='\t')
test = pd.read_csv(root_path + '/data/test_clean.tsv', sep='\t')
print("读取数据完成")

# 将df中的label映射为数字标签并保存到labelIndex列中
labelName = train.label.unique()  # 全部label列表
labelIndex = list(range(len(labelName)))  # 全部label标签
labelNameToIndex = dict(zip(labelName, labelIndex))  # label的名字对应标签的字典
labelIndexToName = dict(zip(labelIndex, labelName))  # label的标签对应名字的字典
# 将训练集中的label名字映射到标签并保存到列labelIndex中
train["labelIndex"] = train.label.map(labelNameToIndex)
# 将测试集中的label名字映射到标签并保存到列labelIndex中
# dev["labelIndex"] = dev.label.map(labelNameToIndex)
# 将测试集中的label名字映射到标签并保存到列labelIndex中
test["labelIndex"] = test.label.map(labelNameToIndex)

# 将query分词并保存在queryCut列中


def query_cut(query):
    return query.split(' ')


train["queryCut"] = train["text"].apply(query_cut)
# dev["queryCut"] = dev["text"].apply(query_cut)
test["queryCut"] = test["text"].apply(query_cut)
print("切分数据完成")
# 读取停用词
with open(root_path + '/data/stopwords.txt', "r") as f:
    stopWords = f.read().split("\n")

# 去除停用词并将去除停用词后的词列表保存到queryCutRMStopWord列中


def rm_stop_word(wordList):
    return [word for word in wordList if word not in stopWords]


train["queryCutRMStopWord"] = train["queryCut"].apply(rm_stop_word)
# dev["queryCutRMStopWord"] = dev["text"].apply(rm_stop_word)
test["queryCutRMStopWord"] = test["queryCut"].apply(rm_stop_word)
print("去除停用词")


def TF_Idf():
    # # 获得所有词组成的列表
    allWords = [word for query in train.queryCutRMStopWord for word in query]
    freWord = Counter(allWords)  # 统计词频，一个字典，键为词，值为词出现的次数
    # 首先计算高频词
    highFreWords = [word for word in freWord.keys() if freWord[word]
                    > 3]  # 词频超过3的词列表

    # 去除低频词函数,训练词库中小于2的词将被抛弃。

    def rm_low_fre_word(query):
        return [word for word in query if word in highFreWords]

    # 去除低频词
    train["queryFinal"] = train["queryCutRMStopWord"].apply(rm_low_fre_word)
    # dev["queryFinal"] = dev["queryCutRMStopWord"] #.apply(rm_low_fre_word)
    test["queryFinal"] = test["queryCutRMStopWord"].apply(rm_low_fre_word)
    print("去除低频词")

    Train_features = em.tfidf.transform(train["queryFinal"]).toarray()
    Test_features = em.tfidf.transform(test["queryFinal"]).toarray()
    print(type(Train_features), Train_features)
    Train_label = train["labelIndex"]
    Test_label = test["labelIndex"]
    print("计算TF-idf")
    print("Train_features.shape =", Train_features.shape)
    print("Test_features.shape =", Test_features.shape)
    print("Train_label.shape =", Train_label.shape)
    print("Test_label.shape =", Test_label.shape)

    return Train_features, Test_features, Train_label, Test_label


################################################################################################
def sentence2vec(query):
    arr = []
    arr = np.array([w2v.wv.get_vector(s)
                    for s in query if s in w2v.wv.vocab.keys()])
    if len(arr) > 0:
        # 第一种方法对一条样本中的词求平均
        return np.mean(np.array(arr), axis=0)
        # 第二种方法返回一条样本中的最大值
        # return np.max(np.array(arr),axis=0)
    else:
        return np.zeros(300)


def Find_Embedding():
    # 生成训练集与测试集，其中x为句子的向量，y采用onehot形式
    # 对词向量进行归一化处理
    min_max_scaler = preprocessing.MinMaxScaler()
    Train_features2 = min_max_scaler.fit_transform(
        np.vstack(train["queryCutRMStopWord"].apply(sentence2vec)))
    Test_features2 = min_max_scaler.fit_transform(
        np.vstack(test["queryCutRMStopWord"].apply(sentence2vec)))

    Train_label2 = train["labelIndex"]
    Test_label2 = test["labelIndex"]

    print("加载训练好的词向量")

    print("Train_features.shape =", Train_features2.shape)
    print("Test_features.shape =", Test_features2.shape)
    print("Train_label.shape =", Train_label2.shape)
    print("Test_label.shape =", Test_label2.shape)

    return Train_features2, Test_features2, Train_label2, Test_label2


def Train_and_Test(Train_features, Test_features, Train_label, Test_label):
    # 构建训练模型并训练及预测
    Embedding_flag = "Word2vec_embedding"
    models = [
        # RandomForestClassifier(
        #     n_estimators=500, max_depth=5, random_state=0),
        # LogisticRegression(solver='liblinear', random_state=0),
        # MultinomialNB(),
        # SVC(),
       lgb.LGBMClassifier(objective='multiclass', n_jobs=10, num_class=33, num_leaves=30, reg_alpha=10, reg_lambda=200,
                           max_depth=3, learning_rate=0.05, n_estimators=2000, bagging_freq=1, bagging_fraction=0.8, feature_fraction=0.8),
    ]
    # 遍历模型
    for model in models:

        model_name = model.__class__.  __name__
        clf = model.fit(Train_features, Train_label)
        print("使用的词嵌入类型{},使用模型:{}".format(embedding_flag, model_name))
        Test_predict_label = clf.predict(Test_features)
        Train_predict_label = clf.predict(Train_features)
        # 输出训练集的准确率
        print(Embedding_flag+"_"+model_name+'_'+'Train accuracy %s' % metrics.accuracy_score(
            Train_label, Train_predict_label))
        logger.info(Embedding_flag+"_"+model_name+'_'+'Train accuracy %s' % metrics.accuracy_score(
            Train_label, Train_predict_label))
        # 输出测试集的准确率
        print(Embedding_flag+"_"+model_name+'_'+'test accuracy %s' % metrics.accuracy_score(
            Test_label, Test_predict_label))
        logger.info(Embedding_flag+"_"+model_name+'_'+'test accuracy %s' % metrics.accuracy_score(
            Test_label, Test_predict_label))
        # 输出recall
        print(Embedding_flag+"_"+model_name+'_'+'test recall %s' % metrics.recall_score(
            Test_label, Test_predict_label, average='micro'))
        logger.info(Embedding_flag+"_"+model_name+'_'+'test recall %s' % metrics.recall_score(
            Test_label, Test_predict_label, average='micro'))
        # 输出F1-score
        # print(Embedding_flag+"_"+model_name+'_'+'test F1_score %s' % metrics.f1_score(Test_label,
        #                                                                               Test_predict_label, average='weighted'))
        # logger.info(Embedding_flag+"_"+model_name+'_'+'test F1_score %s' % metrics.f1_score(Test_label,
        #                                                                                     Test_predict_label ,average='weighted'))
        # 输出精确率
        print(Embedding_flag+"_"+model_name+'_'+'test precision_score %s' % metrics.precision_score(
            Test_label, Test_predict_label, average='micro'))
        logger.info(Embedding_flag+"_"+model_name+'_'+'test precision_score %s' % metrics.precision_score(
            Test_label, Test_predict_label, average='micro'))
        # 输出模型预测错误的类别信息
        # 找出该模型分类错误的样本下标
        predict_error_list = np.argwhere(
            np.array(Test_predict_label-Test_label) != 0)
        # 输出预测错误的前100个样本信息
        # # 每个类别输出5个错误的样本，Dcit02用于计数。
        # Dict02= {}
        # count_number =0
        # for k in range(len(predict_error_list)):
        #     if int(Test_predict_label[predict_error_list[k]]) not in Dict02.keys():
        #         Dict02[int(Test_predict_label[predict_error_list[k]])]=list(predict_error_list[k])
        #         count_number=count_number+1
        #     else:
        #         if len(Dict02[int(Test_predict_label[predict_error_list[k]])]) < 5:
        #             Dict02[int(Test_predict_label[predict_error_list[k]])].append(predict_error_list[k])
        #             count_number=count_number+1
        #         else:
        #             continue

        #     logger.info("预测错误样本的text{},预测标签:{},样本的真实标签:{}".format(np.array(test["queryCutRMStopWord"])[predict_error_list[k]],
        #                                                           labelIndexToName[int(Test_predict_label[predict_error_list[k]])], labelIndexToName[int(Test_label[predict_error_list[k]])]))

        #     if count_number>= 5*len(labelIndex):# 即每个类别都输出了五个预测错误的样本
        #         break;

        # 保存训练好的模型
        joblib.dump(model, root_path +
                    '/src/ML/Saved_ML_Models/'+Embedding_flag+"_"+model_name+'.pkl')


if __name__ == "__main__":
    # TD_IDF
    # Train_features, Test_features, Train_label, Test_label=TF_Idf()
    # Train_and_Test(Train_features=Train_features, Test_features=Test_features,
    #                Train_label=Train_label, Test_label=Test_label)
    # embedding
    Train_features3, Test_features3, Train_label3, Test_label3 = Find_Embedding()
    Train_and_Test(Train_features=Train_features3, Test_features=Test_features3,
                   Train_label=Train_label3, Test_label=Test_label3)
