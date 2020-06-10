import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
# -*- coding: utf-8 -*-
from __init__ import *
from src.utils import config
from src.utils.tools import create_logger, wam
from src.Embedding.embedding import Embedding
from sklearn.externals import joblib
import joblib
from src.utils.tools import Translate_array
from src.utils.config import root_path

logger = create_logger(
    config.log_dir + 'sklearn_models_TDIDF_embeddings6000.log')

# em = Embedding()  # 创建embedding类的对象
# em.load()

Embedding_flag = "TF-IDF6000"
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
print(type(train["queryCutRMStopWord"]))


def TF_Idf():
    # # # 获得所有词组成的列表
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
    print(type(train["queryFinal"]))

    trainText = [' '.join(query) for query in train["queryCutRMStopWord"]]
    testText = [' '.join(query) for query in test["queryCutRMStopWord"]]
    print(type(trainText))
    # AllText = trainText+testText
    # 计算tf-idf值
    # vectorizer = CountVectorizer()  # 获取词袋模型中的所有词语
    stopWords = open(root_path + '/data/stopwords.txt').readlines()
    vectorizer = TfidfVectorizer(
        stop_words=stopWords,  max_df=0.4, min_df=0.001, ngram_range=(1, 2))

    Train_features = vectorizer.fit_transform(
        trainText)  # .toarray() # 转成稠密矩阵太大，运行不了。
    Test_features = vectorizer.transform(testText)
    print(type(Train_features), type(Test_features))
    # 生成训练集与测试集，其中x为所计算的tfidf值

############################################################
    Train_label = train["labelIndex"]
    Test_label = test["labelIndex"]
    print("计算TF-idf")
    print("Train_features.shape =", Train_features.shape)
    print("Test_features.shape =", Test_features.shape)
    print("Train_label.shape =", Train_label.shape)
    print("Test_label.shape =", Test_label.shape)

    return Train_features, Test_features, Train_label, Test_label

#########################################




def Predict(model_name,Train_label, Test_label, Train_predict_label, Test_predict_label):
    # 输出训练集的准确率
    print(Embedding_flag+"_"+model_name+'_'+'Train accuracy %s' % metrics.accuracy_score(
        Train_label, Train_predict_label))
    logger.info(Embedding_flag+"_"+model_name+'_'+'Train accuracy %s' %
     metrics.accuracy_score(Train_label, Train_predict_label))
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
    print(Embedding_flag+"_"+model_name+'_'+'test F1_score %s' %
     metrics.f1_score(Test_label,Test_predict_label, average='weighted'))
    logger.info(Embedding_flag+"_"+model_name+'_'+'test F1_score %s' %
     metrics.f1_score(Test_label,Test_predict_label ,average='weighted'))
    # 输出精确率
    print(Embedding_flag+"_"+model_name+'_'+'test precision_score %s' % metrics.precision_score(
        Test_label, Test_predict_label, average='micro'))
    logger.info(Embedding_flag+"_"+model_name+'_'+'test precision_score %s' % metrics.precision_score(
        Test_label, Test_predict_label, average='micro'))

    predict_error_list = np.argwhere(
        np.array(Test_predict_label-Test_label) != 0)
    # 输出预测错误的前100个样本信息
    print("使用的词嵌入类型{},使用模型:{}".format("TFidf_embedding", model_name))
    # 每个类别输出5个错误的样本，Dcit02用于计数。
    Dict02 = {}
    count_number = 0
    for k in range(len(predict_error_list)):
        if int(Test_predict_label[predict_error_list[k]]) not in Dict02.keys():
            Dict02[int(Test_predict_label[predict_error_list[k]])
                   ] = list(predict_error_list[k])
            count_number = count_number+1
        else:
            if len(Dict02[int(Test_predict_label[predict_error_list[k]])]) < 5:
                Dict02[int(Test_predict_label[predict_error_list[k]])].append(
                    predict_error_list[k])
                count_number = count_number+1
            else:
                continue

        logger.info("预测错误样本的text{},预测标签:{},样本的真实标签:{}".format(np.array(test["queryCutRMStopWord"])[predict_error_list[k]],
        labelIndexToName[int(Test_predict_label[predict_error_list[k]])], labelIndexToName[int(Test_label[predict_error_list[k]])]))

        if count_number >= 5*len(labelIndex):  # 即每个类别都输出了五个预测错误的样本
            break


def Train_and_Test(Train_features, Test_features, Train_label, Test_label):
    # 构建训练模型并训练及预测
    Embedding_flag = "TF-IDF_"
    models = [
        RandomForestClassifier(
            n_estimators=50, max_depth=10, random_state=0),
        LogisticRegression(solver='liblinear', random_state=0),
        MultinomialNB(),
        SVC(),
        lgb.LGBMClassifier(objective='multiclass', n_jobs=10, num_class=33, num_leaves=50, lambda_l1=5, lambda_l2=5, reg_alpha=5, reg_lambda=5,
                           max_depth=5, learning_rate=0.05, n_estimators=500, bagging_freq=5, bagging_fraction=0.8, feature_fraction=0.8),
    ]
    # 遍历模型
    for model in models:
        model_name = model.__class__.  __name__
        print(model_name)
        clf = model.fit(Train_features, Train_label)
        Test_predict_label = clf.predict(Test_features)
        Train_predict_label = clf.predict(Train_features)
        Predict(model_name,Train_label, Test_label,
                Train_predict_label, Test_predict_label)
    #############################################################
    # # 保存训练好的模型
    joblib.dump(model, root_path +
                '/src/ML/Saved_ML_Models/'+Embedding_flag+"_"+model_name+'.pkl')


if __name__ == "__main__":
    # TD_IDF
    Train_features, Test_features, Train_label, Test_label = TF_Idf()
    Train_and_Test(Train_features=Train_features, Test_features=Test_features,
                   Train_label=Train_label, Test_label=Test_label)