#!/usr/local/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from numpy import arange
from matplotlib import pyplot
from pandas import read_csv
from pandas import  set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


def testHouse():
    # 读取房屋数据集
    data = pd.read_csv("house_data.csv")
    # 通过 head 方法查看数据集的前几行数据
    set_option('display.column_space', 120)
    # print(df.head())

    # 数据维度
    print(data.shape)

    # 特征属性的字段类型
    # print(data.dtypes)

    #检查有没有数据中有没有空值
    print(data.isnull().any().sum())

    # 描述性统计信息
    # set_option('precision', 1)
    # print(data.describe())

    #提取特征和标记
    prices = data['MEDV']
    features = data.drop('MEDV', axis=1)

    # 关联关系
    # set_option('precision', 2)
    # print(data.corr(method='pearson'))

    # 直方图
    # data.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
    # pyplot.show()

    # 密度图
    # data.plot(kind='density', subplots=True, layout=(4, 4), sharex=False, fontsize=1)
    # pyplot.show()

    # 箱线图
    # data.plot(kind='box', subplots=True, layout=(4, 4), sharex=False, sharey=False, fontsize=8)
    # pyplot.show()

    # 查看各个特征的散点分布
    scatter_matrix(data, alpha=0.7, figsize=(10, 10), diagonal='kde')
    pyplot.show()


    # Heatmap



def featureSelection():
    data = pd.read_csv("house_data.csv")
    x = data[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
              'PTRATIO', 'B', 'LSTAT']]
    # print(x.head())
    y = data['MEDV']
    from sklearn.feature_selection import SelectKBest
    SelectKBest = SelectKBest(f_regression, k=3)
    bestFeature = SelectKBest.fit_transform(x, y)
    SelectKBest.get_support(indices=False)

    # print(SelectKBest.transform(x))
    print(x.columns[SelectKBest.get_support(indices=False)])

    features = data[['RM', 'PTRATIO', 'LSTAT']].copy()
    # pd.plotting.scatter_matrix(features, alpha=0.7, figsize=(6, 6), diagonal='hist')
    # pyplot.show()

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    for feature in features.columns:
        features.loc[:, '标准化' + feature] = scaler.fit_transform(features[[feature]])

    # 散点可视化，查看特征归一化后的数据
    font = {
        'family': 'SimHei'
    }
    # pyplot.rc('font', **font)
    # pd.plotting.scatter_matrix(features[['标准化RM', '标准化PTRATIO', '标准化LSTAT']], alpha=0.7, figsize=(6, 6),diagonal='hist')
    # pyplot.show()

    #数据集拆分
    x_train, x_test, y_train, y_test = train_test_split(features[['标准化RM', '标准化PTRATIO', '标准化LSTAT']], y,
                                                        test_size=0.3, random_state=33)
    #random_state 表示是否随机划分训练集与测试集，若ransom_state=0，则会随机划分测试集与训练集。随机划分的结果就是会使每次训练的分数不同，程序每运行一次，训练分数就会变化。
    #若使random_state =1(或其他非零数)，则无论程序运行多少次，分数都是相同的。

    #lr
    import warnings
    warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd") #过滤告警

    lr = LinearRegression()
    lr_predict = cross_val_predict(lr, x_train, y_train, cv=5)
    lr_score = cross_val_score(lr, x_train, y_train, cv=5)
    lr_meanscore = lr_score.mean()
    # print(lr_score)
    # print(lr_meanscore)

    #SVR
    from sklearn.svm import SVR
    linear_svr = SVR(kernel = 'linear')
    linear_svr_predict = cross_val_predict(linear_svr, x_train, y_train, cv=5)
    linear_svr_score = cross_val_score(linear_svr, x_train, y_train, cv=5)
    linear_svr_meanscore = linear_svr_score.mean()

    poly_svr = SVR(kernel = 'poly')
    poly_svr_predict = cross_val_predict(poly_svr, x_train, y_train, cv=5)
    poly_svr_score = cross_val_score(poly_svr, x_train, y_train, cv=5)
    poly_svr_meanscore = poly_svr_score.mean()

    rbf_svr = SVR(kernel = 'rbf')
    rbf_svr_predict = cross_val_predict(rbf_svr, x_train, y_train, cv=5)
    rbf_svr_score = cross_val_score(rbf_svr, x_train, y_train, cv=5)
    rbf_svr_meanscore = rbf_svr_score.mean()

    #knn
    # from sklearn.neighbors import KNeighborsRegressor
    # score = []
    # for n_neighbors in range(1, 21):
    #     knn = KNeighborsRegressor(n_neighbors, weights='uniform')
    #     knn_predict = cross_val_predict(knn, x_train, y_train, cv=5)
    #     knn_score = cross_val_score(knn, x_train, y_train, cv=5)
    #     knn_meanscore = knn_score.mean()
    #     score.append(knn_meanscore)
    # plt.plot(score)
    # plt.xlabel('n-neighbors')
    # plt.ylabel('mean-score')
    # plt.show()


    #Decision Tree
    # from sklearn.tree import DecisionTreeRegressor
    # score=[]
    # for n in range(1,11):
    #     dtr = DecisionTreeRegressor(max_depth = n)
    #     dtr_predict = cross_val_predict(dtr, x_train, y_train, cv=5)
    #     dtr_score = cross_val_score(dtr, x_train, y_train, cv=5)
    #     dtr_meanscore = dtr_score.mean()
    #     score.append(dtr_meanscore)
    # plt.plot(np.linspace(1,10,10), score)
    # plt.xlabel('max_depth')
    # plt.ylabel('mean-score')
    # plt.show()

    knn = KNeighborsRegressor(2, weights='uniform')
    knn_predict = cross_val_predict(knn, x_train, y_train, cv=5)
    knn_score = cross_val_score(knn, x_train, y_train, cv=5)
    knn_meanscore = knn_score.mean()

    dtr = DecisionTreeRegressor(max_depth=4)
    dtr_predict = cross_val_predict(dtr, x_train, y_train, cv=5)
    dtr_score = cross_val_score(dtr, x_train, y_train, cv=5)
    dtr_meanscore = dtr_score.mean()

    evaluating = {
        'lr': lr_score,
        'linear_svr': linear_svr_score,
        'poly_svr': poly_svr_score,
        'rbf_svr': rbf_svr_score,
        'knn': knn_score,
        'dtr': dtr_score
    }
    evaluating = pd.DataFrame(evaluating)
    print(evaluating)




def main():
    # testHouse()
    featureSelection()




if __name__ == "__main__":
    main()