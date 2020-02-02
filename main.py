#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:Lijiacai
Email:1050518702@qq.com
===========================================
CopyRight@JackLee.com
===========================================
"""

import os
import sys
import json
import numpy
import pandas
from sklearn import model_selection, tree, ensemble
from sklearn.linear_model import Ridge, RidgeCV, LassoCV, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn import neighbors
from sklearn.metrics import accuracy_score


class DataPreProcessing():
    X_test = None
    X_train = None
    Y_train = None
    Y_test = None

    def __init__(self, dataset):
        self.dataset = dataset

    def data_convert(self, new_column: str, old_column: str, func, drop_old=False):
        """
        将某列数据处理
        :param new_column:  处理后数据的列名
        :param old_column:  旧数据列名
        :param func:
        :return:
        """
        self.dataset[new_column] = eval("self.dataset.{}.apply".format(old_column))(func=func)
        if drop_old:
            self.drop_columns(columns=[old_column])

    def drop_columns(self, columns: list, **kwargs):
        """
        删除多列数据
        :param columns:
        :return:
        """
        self.dataset = self.dataset.drop(columns=columns, **kwargs)

    def data_split_by_columns(self, X_columns: list, Y_column: str, **kwargs):
        self.X_train, self.X_test, self.Y_train, self.Y_test = model_selection.train_test_split(
            self.dataset[pandas.Series(X_columns)], self.dataset[pandas.Series(Y_column)], **kwargs)

    def data_split_by_index(self, start, end, step, Y_column: str, **kwargs):
        self.X_train, self.X_test, self.Y_train, self.Y_test = model_selection.train_test_split(
            self.dataset[self.dataset.columns[start, end, step]], self.dataset[pandas.Series(Y_column)], **kwargs)


class Predictor():
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

    def _output_(self, type_, predict, **kwargs):
        self.rate = mean_squared_error(self.Y_test, predict)
        print("{} mean_squared_error:\n".format(type_), self.rate)

    def model(self, estimator_func=None, estimator_params=None, CV_func=RidgeCV, KN_K=None,
              print_model_info=False, CV_params=None,
              **kwargs):
        """
        获取训练模型
        :param estimator_func: 评估器对象 object
        :param estimator_params: 评估器函数携带参数 dict
        :param CV_func: 多重交叉验证对象 object
        :param KN_K:  K近邻K参数列表 list
        :param print_model_info: 是否输出模型信息 bool
        :param CV_params: 多重交叉验证参数 dict
        :param kwargs:
        :return: 模型对象 object
        """
        if estimator_params == None:
            estimator_params = {}
        if CV_params == None:
            CV_params = {}
        if KN_K == None:
            KN_K = []
        if CV_func in (RidgeCV, LassoCV):
            model = CV_func(**CV_params)
        elif CV_func == GridSearchCV:
            model = GridSearchCV(estimator=estimator_func(**estimator_params), **CV_params)
        elif KN_K:
            cv_result = []
            for k in KN_K:
                cv_knn = model_selection.cross_val_score(estimator=estimator_func(n_neighbors=k, **estimator_params),
                                                         X=self.X_train, y=self.Y_train, **CV_params)
                cv_result.append(cv_knn.mean())
            best_index = numpy.array(cv_result).argmax()
            model = estimator_func(n_neighbors=KN_K[best_index], **estimator_params)
        else:
            raise ("请给多重交叉验证方法RidgeCV/LassoCV/GridSearchCV/,如使用K近邻算法，请给K值")
        model.fit(self.X_train, self.Y_train)
        if print_model_info:
            pred = model.predict(self.X_test)
            self._output_(type_="{}".format(estimator_func.__name__), predict=pred)
        return model


class Classifier(Predictor):

    def _output_(self, type_, predict, **kwargs):
        self.rate = accuracy_score(self.Y_test, predict)
        print("{} accuracy_score:\n".format(type_), self.rate)


class Plot():
    def __init__(self):
        pass


class Modeler():
    Ridge_Params = {}
    Lasso_Params = {}
    DecisionTree_Params = {}
    RandomForest_Params = {}
    KNN_Params = {}
    GridSearch_Params = {}

    def __init__(self, X_train, X_test, Y_train, Y_test, _type_=Predictor):
        self.rate = {}
        if _type_ == Predictor:
            for p in [self.Ridge_Params, self.Lasso_Params, self.DecisionTree_Params, self.RandomForest_Params]:
                if p:
                    self.modeler = _type_(X_train, X_test, Y_train, Y_test)
                    m = self.modeler.model(print_model_info=True, **p)
                    self.rate[p.get("estimator_func").__name__] = m.rate
