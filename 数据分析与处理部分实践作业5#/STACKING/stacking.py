# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 13:58:31 2017

@author: yuwei
"""

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import BayesianRidge
import random


def loadData(fileName):
    "获取数据集"
    #用pandas读入csv文件
    data = pd.read_csv(fileName,encoding='utf-8')
    return data

def spiltTrainTestData(data):
    "划分训练集和测试集"
    train_X, test_X = train_test_split(data,test_size = 0.1,random_state = 0)
    #对随机选择的训练集和测试集重新建立索引
    train_X.index = range(len(train_X))
    test_X.index = range(len(test_X))
    return train_X,test_X
    
def DecisionTree(train_X,test_X):
    "调用决策树回归"
    #训练样本标签为act_class
    train_y = train_X['class'].values
    #训练样本,需要drop掉标签class，即类别标签不加入模型的训练
    train_x = train_X.drop(['class'], axis=1).values
    #测试样本，需要drop掉标签class
    test_x = test_X.drop(['class'], axis=1).values                  
    "模型1"
    #决策树回归进行预测
    clf1 = tree.DecisionTreeRegressor(min_samples_leaf=2,max_depth=100,random_state=2)
    clf1 = clf1.fit(train_x, train_y)
    #获得预测结果
    test1 = clf1.predict(test_x)
    train1 = clf1.predict(train_x)
    #转为DataFrame类型
    test1 = pd.DataFrame(test1)
    train1 = pd.DataFrame(train1)
    "模型2"
    #决策树回归进行预测
    clf2 = tree.DecisionTreeRegressor(min_samples_leaf=2,max_depth=30,random_state=10)
    clf2 = clf2.fit(train_x, train_y)
    #获得预测结果
    test2 = clf2.predict(test_x)
    train2 = clf2.predict(train_x)
    #转为DataFrame类型
    test2 = pd.DataFrame(test2)
    train2 = pd.DataFrame(train2)

    #合并两个模型
    test = pd.concat([test1,test2],axis=1)
    train = pd.concat([train1,train2],axis=1)
    
    print('------决策树回归单模型结果-------')
    print("Mean squared error: %.2f"
      % mean_squared_error(test_X['class'], test1))
    print('Variance score: %.2f' % r2_score(test_X['class'],test1))
    print("Mean squared error: %.2f"
      % mean_squared_error(test_X['class'], test2))
    print('Variance score: %.2f' % r2_score(test_X['class'],test2))
    print('\n')
    return test,train
        
def model_BayesianRidge(train_X,test_X):
    "调用贝叶斯回归"
    #训练样本标签为act_class
    train_y = train_X['class'].values
    #训练样本,需要drop掉标签class，即类别标签不加入模型的训练
    train_x = train_X.drop(['class'], axis=1).values
    #测试样本，需要drop掉标签class
    test_x = test_X.drop(['class'], axis=1).values    
    "模型1"
    #贝叶斯回归进行预测
    clf1 = BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False, copy_X=True,
       fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06, n_iter=300,
       normalize=False, tol=0.001, verbose=False)
    clf1 = clf1.fit(train_x, train_y)
    #获得预测结果
    test1 = clf1.predict(test_x)
    train1 = clf1.predict(train_x)
    #转为DataFrame类型
    test1 = pd.DataFrame(test1)
    train1 = pd.DataFrame(train1)
    "模型2"
    #贝叶斯回归进行预测
    clf2 = BayesianRidge(alpha_1=1e-05, alpha_2=1e-05, compute_score=False, copy_X=True,
       fit_intercept=True, lambda_1=1e-05, lambda_2=1e-05, n_iter=400,
       normalize=False, tol=0.001, verbose=False)
    clf2 = clf2.fit(train_x, train_y)
    #获得预测结果
    test2 = clf2.predict(test_x)
    train2 = clf2.predict(train_x)
    #转为DataFrame类型
    test2 = pd.DataFrame(test2)
    train2 = pd.DataFrame(train2)
    #合并两个模型
    test = pd.concat([test1,test2],axis=1)
    train = pd.concat([train1,train2],axis=1)

    print('------贝叶斯回归单模型结果-------')
    print("Mean squared error: %.2f"
      % mean_squared_error(test_X['class'], test1))
    print('Variance score: %.2f' % r2_score(test_X['class'],test1))
    print("Mean squared error: %.2f"
      % mean_squared_error(test_X['class'], test2))
    print('Variance score: %.2f' % r2_score(test_X['class'],test2))
    print('\n')
    return test,train

def model_Ridge(train_X,test_X):
    "调用Ridge回归"
    #训练样本标签为act_class
    train_y = train_X['class'].values
    #训练样本,需要drop掉标签class，即类别标签不加入模型的训练
    train_x = train_X.drop(['class'], axis=1).values
    #测试样本，需要drop掉标签class
    test_x = test_X.drop(['class'], axis=1).values    
    "模型1"
    #Ridge回归进行预测
    clf1 = Ridge(alpha = 1.5)
    clf1 = clf1.fit(train_x, train_y)
    #获得预测结果
    test1 = clf1.predict(test_x)
    train1 = clf1.predict(train_x)
    #转为DataFrame类型
    test1 = pd.DataFrame(test1)
    train1 = pd.DataFrame(train1)
    "模型2"
    #Ridge回归进行预测
    clf2 = Ridge(alpha = .5)
    clf2 = clf2.fit(train_x, train_y)
    #获得预测结果
    test2 = clf2.predict(test_x)
    train2 = clf2.predict(train_x)
    #转为DataFrame类型
    test2 = pd.DataFrame(test2)
    train2 = pd.DataFrame(train2)
    #合并两个模型
    test = pd.concat([test1,test2],axis=1)
    train = pd.concat([train1,train2],axis=1)
    
    print('-------Ridge回归单模型结果--------')
    print("Mean squared error: %.2f"
      % mean_squared_error(test_X['class'], test1))
    print('Variance score: %.2f' % r2_score(test_X['class'],test1))
    print("Mean squared error: %.2f"
      % mean_squared_error(test_X['class'], test2))
    print('Variance score: %.2f' % r2_score(test_X['class'],test2))
    print('\n')
    return test,train
    
def model_GBR(train_X,test_X):
    "调用背景梯度提升回归"
    #训练样本标签为act_class
    train_y = train_X['class'].values
    #训练样本,需要drop掉标签class，即类别标签不加入模型的训练
    train_x = train_X.drop(['class'], axis=1).values
    #测试样本，需要drop掉标签class
    test_x = test_X.drop(['class'], axis=1).values    
    "模型1"
    #GBR回归进行预测
    clf1 = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0,max_depth=2, random_state=0, loss='ls')
    clf1 = clf1.fit(train_x, train_y)
    #获得预测结果
    test1 = clf1.predict(test_x)
    train1 = clf1.predict(train_x)
    #转为DataFrame类型
    test1 = pd.DataFrame(test1)
    train1 = pd.DataFrame(train1)
    "模型2"
    #GBR回归进行预测
    clf2 = GradientBoostingRegressor(n_estimators=100, learning_rate=1.5,max_depth=1, random_state=0)
    clf2 = clf2.fit(train_x, train_y)
    #获得预测结果
    test2 = clf2.predict(test_x)
    train2 = clf2.predict(train_x)
    #转为DataFrame类型
    test2 = pd.DataFrame(test2)
    train2 = pd.DataFrame(train2)
    #合并两个模型
    test = pd.concat([test1,test2],axis=1)
    train = pd.concat([train1,train2],axis=1)
    
    print('--------GRB回归单模型结果---------')
    print("Mean squared error: %.2f"
      % mean_squared_error(test_X['class'], test1))
    print('Variance score: %.2f' % r2_score(test_X['class'],test1))
    print("Mean squared error: %.2f"
      % mean_squared_error(test_X['class'], test2))
    print('Variance score: %.2f' % r2_score(test_X['class'],test2))
    print('\n')
    return test,train
    
def model_RFR(train_X,test_X):
    "特征的随机"
    "调用随机森林提升回归"
    #存储标签
    attribute = list(train_X.columns[:-1])
    #存储标签列
    label_tr = train_X[train_X.columns[-1]]
    label_te = test_X[test_X.columns[-1]]
    #每次特征选择特征个数，每次只取特征总数的80%进行训练
    numAttribute = int(len(attribute)*0.8)
    #根据特征个数随机选特征
    randomAttribute = random.sample(attribute, numAttribute)
    #选择对应特征的属性列
    #训练集
    train_X = train_X[randomAttribute]
    train_X = pd.concat([train_X,label_tr],axis=1)
    #测试集
    test_X = test_X[randomAttribute]
    test_X = pd.concat([test_X,label_te],axis=1)
    
    #训练样本标签为act_class
    train_y = train_X['class'].values
    #训练样本,需要drop掉标签class，即类别标签不加入模型的训练
    train_x = train_X.drop(['class'], axis=1).values
    #测试样本，需要drop掉标签class
    test_x = test_X.drop(['class'], axis=1).values    
    "模型1"
    #RFR回归进行预测
    clf1 = RandomForestRegressor(n_estimators=18, max_depth=3,min_samples_split=2, random_state=0)
    clf1 = clf1.fit(train_x, train_y)
    #获得预测结果
    test1 = clf1.predict(test_x)
    train1 = clf1.predict(train_x)
    #转为DataFrame类型
    test1 = pd.DataFrame(test1)
    train1 = pd.DataFrame(train1)

    "模型2"
    #RFR回归进行预测
    clf2 = RandomForestRegressor(n_estimators=20, max_depth=2,min_samples_split=2, random_state=0)
    clf2 = clf2.fit(train_x, train_y)
    #获得预测结果
    test2 = clf2.predict(test_x)
    train2 = clf2.predict(train_x)
    #转为DataFrame类型
    test2 = pd.DataFrame(test2)
    train2 = pd.DataFrame(train2)
    #合并两个模型
    test = pd.concat([test1,test2],axis=1)
    train = pd.concat([train1,train2],axis=1)
    
    print('-特征随机后的随机森林回归单模型结果-')
    print("Mean squared error: %.2f"
      % mean_squared_error(test_X['class'], test1))
    print('Variance score: %.2f' % r2_score(test_X['class'],test1))
    print("Mean squared error: %.2f"
      % mean_squared_error(test_X['class'], test2))
    print('Variance score: %.2f' % r2_score(test_X['class'],test2))
    print('\n')
    return test,train
    
def modelMerge(train_X,test_X):
    "模型融合"
    #决策树回归
    testDT,trainDT = DecisionTree(train_X,test_X)
    #贝叶斯回归
    testBayes,trainBayes = model_BayesianRidge(train_X,test_X)
    #Ridge回归
    testRidge,trainRidge = model_Ridge(train_X,test_X)
    #GBR回归
    testGBR,trainGBR = model_GBR(train_X,test_X)
    #RFR回归
    testRFR,trainRFR = model_RFR(train_X,test_X)
    #合并训练集
    train = pd.concat([train_X,trainDT],axis=1)
    train = pd.concat([train,trainBayes],axis=1)
    train = pd.concat([train,trainRidge],axis=1)
    train = pd.concat([train,trainGBR],axis=1)
    train = pd.concat([train,trainRFR],axis=1)
    #合并验证集
    test = pd.concat([test_X,testDT],axis=1)
    test = pd.concat([test,testBayes],axis=1)
    test = pd.concat([test,testRidge],axis=1)
    test = pd.concat([test,testGBR],axis=1)
    test = pd.concat([test,testRFR],axis=1)
    return train,test
    
def modelEvaluate(train_X,test_X,train,test):
    "回归模型评价"

    "线性回归单模型结果"
    #训练样本标签为act_class
    train_y = train_X['class'].values
    #训练样本,需要drop掉标签class，即类别标签不加入模型的训练
    train_x = train_X.drop(['class'], axis=1).values
    #测试样本，需要drop掉标签class
    test_x = test_X.drop(['class'], axis=1).values    
    clf1 = LinearRegression()
    clf1 = clf1.fit(train_x, train_y)
    #获得预测结果
    predict1 = clf1.predict(test_x)
    print('-------线性回归单模型结果--------')
    print("Mean squared error: %.2f"
      % mean_squared_error(test['class'], predict1))
    print('Variance score: %.2f' % r2_score(test['class'],predict1))
    print('\n')
    
    "模型融合结果"
    #训练样本标签为act_class
    train_y = train['class'].values
    #训练样本,需要drop掉标签class，即类别标签不加入模型的训练
    train_x = train.drop(['class'], axis=1).values
    #测试样本，需要drop掉标签class
    test_x = test.drop(['class'], axis=1).values    
    clf1 = LinearRegression()
    clf1 = clf1.fit(train_x, train_y)
    #获得预测结果
    predict1 = clf1.predict(test_x)
    print('----------模型融合结果-----------')
    print("Mean squared error: %.2f"
      % mean_squared_error(test['class'], predict1))
    print('Variance score: %.2f' % r2_score(test['class'],predict1))
    
if __name__ == '__main__': 
    print('评价指标 Mean squared error 越小越好,0是最优\n评价指标  variance score  越接近1越好,1是最优\n')
    #获取数据集
    data = loadData('cpu.csv')
    #划分数据集
    train_X,test_X = spiltTrainTestData(data)
    #得到10个模型融合的训练集和测试集
    train,test = modelMerge(train_X,test_X)
    #调用模型评估函数，比较单模型线性回归和模型融合结果
    modelEvaluate(train_X,test_X,train,test)

    