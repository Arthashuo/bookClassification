import numpy as np
import pandas as pd
import jieba
import requests
from scipy import sparse
import os
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from collections import Counter
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from gensim.models import Word2Vec
from sklearn.svm import SVC
from sklearn.decomposition import PCA
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
from bayes_opt import BayesianOptimization
from __init__ import *
from src.utils import config
from src.utils.tools import create_logger, Translate_array
from src.Pre_process.mlData import MLData
from src.utils.config import root_path
from gensim import models
from hyperopt import fmin, tpe, hp, space_eval, rand, Trials, partial, STATUS_OK
logger = create_logger(
    config.log_dir + 'sklearn_models_feature_engineer.log')
max_length = 500  # 表示样本表示最大的长度

Train_features3, Test_features3, Train_label3, Test_label3 = [], [], [], []
# 加载embedding
fast_embedding = models.KeyedVectors.load(
    root_path + '/src/Embedding/models/fast_model_50000')
w2v_embedding = models.KeyedVectors.load(
    root_path + '/src/Embedding/models/w2v_model_50000')

print("fast_embedding输出词表的个数{},w2v_embedding输出词表的个数{}".format(
    len(fast_embedding.wv.vocab.keys()), len(w2v_embedding.wv.vocab.keys())))


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
    ########################################################################
    trainText = [' '.join(query) for query in train["queryCutRMStopWord"]]
    testText = [' '.join(query) for query in test["queryCutRMStopWord"]]
    stopWords = open(root_path + '/data/stopwords.txt').readlines()
    vectorizer = TfidfVectorizer(
        stop_words=stopWords, max_df=0.5, min_df=0.1, ngram_range=(1, 2))

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

################################################################################################

# 该函数用于在指定窗口大小条件下的词向量,第一个参数为词向量矩阵，第二个参数为滑动窗口大小
# embedding_matrix表示该矩阵形成的词嵌入矩阵，返回滑动窗口2，3，4的最大值和平均值拼接成的词向量


def Find_embedding_with_windows(embedding_matrix):
    # 最终的词向量
    result_list = []
    for window_size in range(2, 5):
        max_list, avg_list = [], []
        for k1 in range(len(embedding_matrix)):
            if int(k1+window_size) > len(embedding_matrix):
                break
            else:
                matrix01 = embedding_matrix[k1:k1+window_size]
                max_list.extend([np.max(matrix01)])  # 最大池化层
                avg_list.extend([np.mean(matrix01)])  # 均值池化层
        # 再将池化层和均值层拼接起来
        max_list.extend(avg_list)
        # 将窗口为2，3，4的embedding拼接起来
        result_list.extend(max_list)
    return result_list


# 获取标签空间的词嵌入
def Find_Label_embedding(example_matrix, embedding):
    # 遍历所有的label,返回标签矩阵
    label_arr = np.array([embedding.wv.get_vector(labelIndexToName[key])
                          for key in labelIndexToName if labelIndexToName[key] in embedding.wv.vocab.keys()])
    similarity_matrix = np.dot(example_matrix, np.transpose(label_arr))
    # 然后对相似矩阵进行均值池化，则得到了“类别-词语”的注意力机制
    similarity_matrix_avg = np.mean(similarity_matrix, axis=1)
    # 将样本的词嵌入与注意力机制相乘得到
    attention_embedding = example_matrix * \
        similarity_matrix_avg.reshape(len(similarity_matrix_avg), 1)
    attention_embedding_avg = np.mean(attention_embedding, axis=0)
    attention_embedding_max = np.max(attention_embedding, axis=0)
    result_embedding = np.hstack(
        (attention_embedding_avg, attention_embedding_max))

    return result_embedding


# 根据多个模型产生的词嵌入来构造新的样本表示


def sentence2vec(query):
    global max_length
    arr = []
    # 加载fast_embedding,w2v_embedding
    global fast_embedding, w2v_embedding
    fast_arr = np.array([fast_embedding.wv.get_vector(s)
                         for s in query if s in fast_embedding.wv.vocab.keys()])
    # 在fast_arr下滑动获取到的词向量
    if len(fast_arr) > 0:
        windows_fastarr = np.array(Find_embedding_with_windows(fast_arr))
        result_attention_embedding = Find_Label_embedding(
            fast_arr, fast_embedding)
    else:
        windows_fastarr = np.zeros(300)
        result_attention_embedding = np.zeros(300)

    # w2v_arr = np.array([w2v_embedding.wv.get_vector(s)
    #                     for s in query if s in w2v_embedding.wv.vocab.keys()])
    # windows_w2varr=Find_embedding_with_windows(w2v_arrs)
    # w2v_arr = np.mean(np.array(w2v_arr), axis=0) if len(
    #     w2v_arr) > 0 else np.zeros(300)

    fast_arr_max = np.max(np.array(fast_arr), axis=0) if len(
        fast_arr) > 0 else np.zeros(300)
    fast_arr_avg = np.mean(np.array(fast_arr), axis=0) if len(
        fast_arr) > 0 else np.zeros(300)

    fast_arr = np.hstack((fast_arr_avg, fast_arr_max))
    # 将多个embedding进行横向拼接
    arr = np.hstack((np.hstack((fast_arr, windows_fastarr)),
                     result_attention_embedding))
    sentence_max_length = 1500  # 表示句子/样本在降维之前的维度
    # 如果样本的维度大于指定的长度则需要进行截取或者拼凑,
    result_arr = arr[:sentence_max_length] if len(arr) > sentence_max_length else np.hstack((
        arr, np.zeros(int(sentence_max_length-len(arr)))))

    #print(type(result_arr),result_arr.shape)
    return result_arr

# 特征选择/抽取函数，对经过特征工程之后高维的特征进行降维
# max_length表示样本最大的维度


def Dimension_Reduction(Train, Test):
    global max_length
    pca = PCA(n_components=max_length)
    pca_train = pca.fit_transform(Train)
    pca_test = pca.fit_transform(Test)

    return pca_train, pca_test



def Find_Embedding():
    # 生成训练集与测试集，其中x为句子的向量，y采用onehot形式
    # 对词向量进行归一化处理
    # 获取样本经过特征工程之后的样本表示，https://mp.weixin.qq.com/s/k-gS6k3-hy-ZI_r901IGvg
    min_max_scaler = preprocessing.MinMaxScaler()
    Train_features2 = min_max_scaler.fit_transform(
        np.vstack(train["queryCutRMStopWord"].apply(sentence2vec)))
    Test_features2 = min_max_scaler.fit_transform(
        np.vstack(test["queryCutRMStopWord"].apply(sentence2vec)))
    # 在对样本表示进行特征选择操作实现降维
    Train_features2, Test_features2 = Dimension_Reduction(
        Train=Train_features2, Test=Test_features2)

    Train_label2 = train["labelIndex"]
    Test_label2 = test["labelIndex"]

    print("加载训练好的词向量")

    print("Train_features.shape =", Train_features2.shape)
    print("Test_features.shape =", Test_features2.shape)
    print("Train_label.shape =", Train_label2.shape)
    print("Test_label.shape =", Test_label2.shape)

    return Train_features2, Test_features2, Train_label2, Test_label2


def Predict(Train_label, Test_label, Train_predict_label, Test_predict_label, model_name):
    # 输出训练集的准确率
    print(model_name+'_'+'Train accuracy %s' % metrics.accuracy_score(
        Train_label, Train_predict_label))
    logger.info(model_name+'_'+'Train accuracy %s' % metrics.accuracy_score(
        Train_label, Train_predict_label))
    # 输出测试集的准确率
    print(model_name+'_'+'test accuracy %s' % metrics.accuracy_score(
        Test_label, Test_predict_label))
    logger.info(model_name+'_'+'test accuracy %s' % metrics.accuracy_score(
        Test_label, Test_predict_label))

# 基于网格搜索


def Grid_Train_model(Train_features, Test_features, Train_label, Test_label):
    # 构建训练模型并训练及预测
    # 网格搜索
    parameters = {
        'max_depth': [5, 10, 15, 20, 25],
        'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
        'n_estimators': [100, 500, 1000, 1500, 2000],
        'min_child_weight': [0, 2, 5, 10, 20],
        'max_delta_step': [0, 0.2, 0.6, 1, 2],
        'subsample': [0.6, 0.7, 0.8, 0.85, 0.95],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
        'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1],
        'scale_pos_weight': [0.2, 0.4, 0.6, 0.8, 1]

    }
    models = [
        lgb.LGBMClassifier(objective='multiclass', n_jobs=10, num_class=33, num_leaves=30, reg_alpha=10, reg_lambda=200,
                           max_depth=3, learning_rate=0.05, n_estimators=2000, bagging_freq=1, bagging_fraction=0.9, feature_fraction=0.8, seed=1440),
    ]
    # 遍历模型
    for model in models:
        model_name = model.__class__.  __name__
        #clf = model.fit(Train_features, Train_label)
        # Test_predict_label = clf.predict(Test_features)
        # Train_predict_label = clf.predict(Train_features)
        # 有了gridsearch我们便不需要fit函数
        gsearch = GridSearchCV(
            model, param_grid=parameters, scoring='accuracy', cv=3)
        gsearch.fit(Train_features, Train_label)
        # 输出最好的参数
        print("Best parameters set found on development set:{}".format(
            gsearch.best_params_))
        ############################################
        Test_predict_label = gsearch.predict(Test_features)
        Train_predict_label = gsearch.predict(Train_features)
        Predict(Train_label, Test_label,
                Train_predict_label, Test_predict_label, model_name)

        # 保存训练好的模型
        joblib.dump(model, root_path +
                    '/src/ML/Saved_ML_Models/'+"feature_engineer_"+model_name+'.pkl')

# 基于贝叶斯优化的搜索


lgb_lgbm = None


def light_GBM(argsDict):
    max_depth = argsDict["max_depth"] + 5
    n_estimators = argsDict['n_estimators'] * 5 + 50
    learning_rate = argsDict["learning_rate"] * 0.02 + 0.05
    subsample = argsDict["subsample"] * 0.1 + 0.7
    min_child_weight = argsDict["min_child_weight"]+1
    global lgb_lgbm
    lgb_lgbm = lgb.LGBMClassifier(nthread=4,  # 进程数
                                  max_depth=max_depth,  # 最大深度
                                  n_estimators=n_estimators,  # 树的数量
                                  learning_rate=learning_rate,  # 学习率
                                  subsample=subsample,  # 采样数
                                  min_child_weight=min_child_weight,  # 孩子数
                                  max_delta_step=10,  # 10步不降则停止
                                  objective='multiclass',
                                  n_jobs=10,
                                  num_class=33,
                                  reg_alpha=10,
                                  reg_lambda=200)

    global Train_features3, Test_features3, Train_label3, Test_label3
    lgb_lgbm.fit(Train_features3, Train_label3)
    predict_Train_labels = lgb_lgbm.predict(Train_features3)
    metric_auc = metrics.accuracy_score(Train_label3, predict_Train_labels)
    # 由于我们是要求auc尽可能的大，但是优化是最小值，因此需要加负号
    return -metric_auc


def Bayes_optimization():
    space = {"max_depth": hp.randint("max_depth", 15),
             # [0,1,2,3,4,5] -> [50,]
             "n_estimators": hp.randint("n_estimators", 1000),
             # [0,1,2,3,4,5] -> 0.05,0.06
             "learning_rate": hp.randint("learning_rate", 6),
             # [0,1,2,3] -> [0.7,0.8,0.9,1.0]
             "subsample": hp.randint("subsample", 4),
             "min_child_weight": hp.randint("min_child_weight", 5),
             }
    best = fmin(light_GBM, space, algo=partial(
        tpe.suggest, n_startup_jobs=1), max_evals=4)

    print(best)
    print(light_GBM(best))

    # 基于贝叶斯优化之后的模型对测试集进行测试
    global Train_features3, Test_features3, Train_label3, Test_label3
    predict_Train_labels = lgb_lgbm.predict(Train_features3)
    predict_Test_labels = lgb_lgbm.predict(Test_features3)
    Predict(Train_label3, Test_label3,
            predict_Train_labels, predict_Test_labels, model_name='bayes_LightGBM')


if __name__ == "__main__":
    # embedding
    Train_features3, Test_features3, Train_label3, Test_label3 = Find_Embedding()
    # 根据网格进行搜索
    Grid_Train_model(Train_features=Train_features3, Test_features=Test_features3,
                     Train_label=Train_label3, Test_label=Test_label3)

    # 根据贝叶斯进行搜索
    # Bayes_optimization()
