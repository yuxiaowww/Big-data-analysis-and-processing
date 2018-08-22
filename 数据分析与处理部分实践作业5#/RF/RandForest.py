# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 17:01:14 2017

@author: yuwei
"""

import pandas as pd
from sklearn import tree
import math
import random
from sklearn.cross_validation import train_test_split


def loadData(fileName):
    "获取数据集"
    data = pd.read_csv(fileName,encoding='utf-8')
    #存储整个数据集
    dataSet = data
    #存储所有属性
    attribute = list(data.columns)[:-1]
    #存储所有属性的值，不含第一行属性名
    dataAttribute = (data.values)
    return dataAttribute,attribute,dataSet
    
def spiltTrainTestData(dataSet):
    "划分训练集和测试集"
    #比例定为8:2
    train_X, test_X = train_test_split(dataSet,test_size = 0.2,random_state = 0)
    #对随机选择的训练集和测试集重新建立索引
    train_X.index = range(len(train_X))
    test_X.index = range(len(test_X))
    return train_X,test_X
  
def randSelectAttribute(attribute,train_data,test_data):
    "随机选择特征,返回特征选择后的结果"
    #存储标签列
    label_tr = train_data[train_data.columns[-1]]
    label_te = test_data[test_data.columns[-1]]
    #每次特征选择特征个数
    numAttribute = math.sqrt(len(attribute))
    #根据特征个数随机选特征,调用random.sample实现随机选特征
    randomAttribute = random.sample(attribute, math.ceil(numAttribute)+3)
    #对训练集和测试集选择对应特征的属性列
    #训练集
    tr = train_data[randomAttribute]
    tr = pd.concat([tr,label_tr],axis=1)
    #测试集
    te = test_data[randomAttribute]
    te = pd.concat([te,label_te],axis=1)
    return tr,te,randomAttribute    
    
def callDecisionTree(train_data,test_data,attribute):
    # 测试样本标签为
    train_y = train_data['class'].values
    train_x = train_data.drop(['class'], axis=1).values
    test_x = test_data.drop(['class'], axis=1).values
    #调用决策树模型
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_x, train_y)
    #获取测试集结果
    predicted_proba = clf.predict(test_x)
    #转为DataFrame类型
    predicted_proba = pd.DataFrame(predicted_proba)
	#加入预测结果
    predict = pd.concat([test_data[attribute], predicted_proba], axis=1)
    return predict
       
def randomForest(n,train_data,test_data):
    "调用n次决策树，并进行投票"
    #存储预测结果
    predict_data = test_data.copy()
    #增加类别列，投票使用
    classSet = list(set(predict_data['class']))
    for i in range(len(classSet)):
        predict_data[classSet[i]] = 0
    #统计每个类别的票数
    for i in range(n):
        print('正在为第'+str(i+1)+'棵树统计票数')
        #随机选特征返回训练集和测试集
        tr,te,randomAttribute = randSelectAttribute(attribute,train_data,test_data)
#        print('此棵树随机特征选择情况:'+str(randomAttribute))
        #返回当前随机特征下调用决策树返回预测结果
        predict = callDecisionTree(tr,te,randomAttribute)
        #统计票数，准备投票
        for i in range(len(predict)):
            predict_data[predict[predict.columns[-1]].loc[i]].loc[i] += 1
    #投票
    predict_data['predict_class'] = 0
    for i in range(len(predict_data)):
        for j in range(len(classSet)):
            #进行投票，选择票数的类别作为最终分类类别
            if max(predict_data[classSet].loc[i]) == predict_data[classSet[j]].loc[i]:
                predict_data['predict_class'].loc[i] = classSet[j]
    return predict_data
    
def DecisionTree(train_X,test_X):
    "调用决策树,返回决策树的预测结果"
    #训练样本标签为act_class
    train_y = train_X['class'].values
    #训练样本,需要drop掉标签class，即类别标签不加入模型的训练
    train_x = train_X.drop(['class'], axis=1).values
    #测试样本，需要drop掉标签class
    test_x = test_X.drop(['class'], axis=1).values
    #决策树分类进行预测
    clf = tree.DecisionTreeClassifier(max_depth=15,min_samples_leaf=2,random_state=1)
    clf = clf.fit(train_x, train_y)
    #获得预测结果
    predicted = clf.predict(test_x)
    #转为DataFrame类型
    predicted = pd.DataFrame(predicted)
    #返回预测结果
    predict = pd.concat([test_X,predicted],axis=1)
    return predict
      
def evaluate(predict):
    "利用F1值评价随机森林和决策树模型"
    #计算TP、FN、FP、TN
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i in range(len(predict)):
        #如果预测对了正样本：TP加1
        if (predict['class'][i] == predict[predict.columns[-1]][i]) & (predict['class'][i] == 'tested_positive'):
            TP += 1
        #如果预测对了负样本：TN加1
        elif (predict['class'][i] == predict[predict.columns[-1]][i]) & (predict['class'][i] == 'tested_negative'):
            TN += 1
        #如果预测错了样本分类，且真实样本为正样本：FN加1
        elif (predict['class'][i] != predict[predict.columns[-1]][i]) & (predict['class'][i] == 'tested_positive'):
            FN += 1
        #如果预测错了样本分类，且真实样本为负样本：FP加1
        elif (predict['class'][i] != predict[predict.columns[-1]][i]) & (predict['class'][i] == 'tested_negative'):
            FP += 1
    #存储查全和查准率
    P = 0
    R = 0
    P = float(TP)/float(TP+FP)
    R = float(TP)/float(TP+FN)
    F1 = float(2*P*R)/(P+R)
    ACC = float(TP+TN)/(TP+TN+FN+FP)
    print('P(查准率)='+str(P) + ' R(查全率)='+str(R) + ' ACC(准确率)='+str(ACC)+' F1='+str(F1))
    
if __name__ == '__main__': 
    data,attribute,dataSet = loadData('diabetes.csv')
    #随机划分数据集
    train_X,test_X = spiltTrainTestData(dataSet)
    print('随机森林模型：')
    predictRF = randomForest(30,train_X,test_X)
    print('----------RF评价-----------')
    evaluate(predictRF)
    print('\n')
    print('决策树模型：')
    predictDT = DecisionTree(train_X,test_X)
    print('----------DT评价-----------')
    evaluate(predictDT)
    

    