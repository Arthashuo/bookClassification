'''
@Author: xiaoyao jiang
@Date: 2020-04-08 19:39:30
@LastEditTime: 2020-04-12 18:22:11
@LastEditors: Please set LastEditors
@Description: use logistic model to classifier
@FilePath: /textClassification/src/ML/logistic.py
'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib
# from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from __init__ import *
from src.utils import config
from src.utils.tools import create_logger
from src.data.mlData import MLData
logger = create_logger(config.log_dir + 'lr.log')


class Models(object):
    def __init__(self):
        self.ml_data = MLData()
        self.models = [
            RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
            LinearSVC(),
            # MultinomialNB(),
            LogisticRegression(random_state=0),
        ]

    def model_select(self, features, labels, CV=5):
        cv_df = pd.DataFrame(index=range(CV * len(self.models)))

        entries = []
        for model in self.models:
            model_name = model.__class__.__name__
            accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
            for fold_idx, accuracy in enumerate(accuracies):
                entries.append((model_name, fold_idx, accuracy))

        cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
        mean_accuracy = cv_df.groupby('model_name').accuracy.mean()
        std_accuracy = cv_df.groupby('model_name').accuracy.std()

        acc = pd.concat([mean_accuracy, std_accuracy], axis=1, ignore_index=True)
        acc.columns = ['Mean Accuracy', 'Standard deviation']
        return cv_df, acc

    def summary(self, y_test, y_pred):
        conf_mat = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt='d',
                    xticklabels=self.ml_data.category_id_df.label.values,
                    yticklabels=self.ml_data.category_id_df.label.values)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title("CONFUSION MATRIX - LinearSVC\n", size=16)

    def predict(self, text, model):
        return model.predict(self.ml_data.em.count_vect.transform([text]))


if __name__ == "__main__":
    m = Models()
    X_train, X_test, y_train, y_test = m.ml_data.process_data(method='tfidf')
    logger.info('model select with tfidf')
    cv_df, acc = m.model_select(X_train, y_train)
    logger.info(cv_df)
    logger.info(acc)

    X_train, X_test, y_train, y_test = m.ml_data.process_data(method='word2vec')
    logger.info('model select with word2vec')
    cv_df, acc = m.model_select(X_train, y_train)
    logger.info(cv_df)
    logger.info(acc)

    X_train, X_test, y_train, y_test = m.ml_data.process_data(method='fasttext')
    logger.info('model select with fasttext')
    cv_df, acc = m.model_select(X_train, y_train)
    logger.info(cv_df)
    logger.info(acc)