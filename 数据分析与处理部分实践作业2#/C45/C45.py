# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 09:58:03 2017

@author: yuwei
"""

import pandas as pd
import math
import operator
import numpy as np

def loadData(fileName):
    "获取数据集"
    data = pd.read_csv(fileName)
    #存储整个数据集
    dataSet = data
    #存储所有属性
    attribute = list(data.columns)[:-1]
    #存储所有属性的值，不含第一行属性名
    dataAttribute = (data.values)
    return dataAttribute,attribute,dataSet

def calculateComentropy(data):
    "计算信息熵"
    '''
    信息熵定义：每个标签计算(-p*log2p),再求和得到整体信息熵
    '''
    #存储信息熵的值
    comentropy = 0
    #计算数据集的样本数量
    sampleSize = len(data)
    #字典存储数据集的标签
    labels = {}
    #循环数据集中的每一个样本
    for sample in data:
        #当前的标签为样本最后一个值
        labelNow = sample[-1]
        #假如当前的标签不在标签字典的键值中，添加进入字典
        if labelNow not in labels.keys():
            labels[labelNow] = 0
        #如果已经存在，在对应位置+1
        labels[labelNow] += 1
    #循环字典中的键值，就是所有的标签
    for key in labels:
        #当前标签个数除以样本数量就是标签概率
        probability = float(labels[key])/float(sampleSize)
        #由信息熵公式计算每个标签信息熵，累加求和
        comentropy = -probability*math.log(probability,2) + comentropy
    #循环结束，返回信息熵计算结果
    return comentropy
    
def spiltDataSet(data,row,rowValue):
    "划分数据集"
    '''
    传入数据集，要分割的列，要分割的列里面要分割的值
    '''
    #存储最终的划分结果
    spiltedData = []
    #循环所有样本
    for sample in data:
        #当前样本的列标签对应值为rowValue，将该样本除row列加入最终结果
        if sample[row] == rowValue:
            #不包含row，即切片跳过row
            reduceSample = list(sample[:row])
            reduceSample.extend(sample[row+1:])
            #将去除row列的数据加入到最终结果
            spiltedData.append(reduceSample)
    return spiltedData
    
def gainBeatAttribute(data):
    "获得最优属性划分--C45 信息增益率"
    '''
    计算信息增益率公式
    增益率 = 信息增益/固有值   固有值 = 每个属性中的值的概率值(-p*log2p) 然后再求和
    '''
    #计算属性的个数
    numAttribute = len(data[0]) - 1
    #调用信息熵函数，计算当前数据的信息熵
    comentropyData = calculateComentropy(data)
    #存储最大的信息增益率
    bestInformationGainRatio = 0
    #存储最大信息增益的属性
    bestAtrribute = -1
    #循环所有属性列，计算每次数据划分后的信息增益，找出信息增益最大的属性
    for i in range(numAttribute):
        #依次取出数据集中的每一列
        attribute = []
        for sample in data:
            attribute.append(sample[i])
        #attribute = [sample[i] for sample in data]
        #转为集合，去重该列的数据值
        uniqueAtrribute = set(attribute)
        #存储数据集划分后的信息熵
        splitedComentropy = 0
        #循环该列数据中的每一个值
        #存储固有值
        intrinsicValue = 0
        for value in uniqueAtrribute:
            #分别按该列数据的每一个值来划分数据
            spiltedData = spiltDataSet(data,i,value)
            #计算数据划分后的数据集的概率
            probability = float(len(spiltedData))/float(len(data))
            #计算当前划分后的数据集的固有值
            intrinsicValue += -probability*math.log(probability,2) 
            #计算当前划分后的信息熵,累加每个值
            splitedComentropy += probability*calculateComentropy(spiltedData)
        #计算按当前划分后的信息增益值
        informationGainRatio = (comentropyData - splitedComentropy)/float(intrinsicValue+0.000001)
        #判断当前的信息增益是否为最大
        if informationGainRatio > bestInformationGainRatio:
            bestInformationGainRatio = informationGainRatio
            #存储当前最优划分属性的列标记
            bestAtrribute = i
    #返回最优划分属性的列
    return bestAtrribute

def majorityCount(classificationLabel):  
    # 定义标签元字典，key为标签，value为标签的数目  
    classificationCount = {}  
    # 遍历所有标签  
    for vote in classificationLabel:  
        #如果标签不在元字典对应的key中  
        if vote not in classificationCount.keys():  
            # 将标签放到字典中作为key，并将值赋为0  
            classificationCount[vote] = 0  
        # 对应标签的数目加1  
        classificationCount[vote] += 1  
    # 对所有标签按数目排序  
    sortedClassificationCount = sorted(classificationCount.iteritems(),key=operator.itemgetter(1),reverse=True)  
    # 返回数目最多的标签  
    return sortedClassificationCount[0][0]
    
def createC45Tree(data,attribute):
    "构建树，传入属性值和属性"
    classificationLabel = []
    for sample in data:
        classificationLabel.append(sample[-1])
    '''类别完全一致则停止划分'''
    if classificationLabel.count(classificationLabel[0]) == len(data):
        return classificationLabel[0]
    '''数据集只有一个样本则停止划分'''
    if len(data) == 1:
        return majorityCount(classificationLabel)
    #获取信息增益率最大的属性
    bestAtrribute = gainBeatAttribute(data)
    # 找到最好的属性  
    bestAtrributeLabel = attribute[bestAtrribute]  
    # 定义决策树，key为bestFeatLabel，value为空  
    C45Tree = {bestAtrributeLabel:{}}  
    # 删除atrribute[bestAtrribute]  对应的元素
    del(attribute[bestAtrribute])  
    # 取出data中bestFeat列的所有值  
    atrributeValues = [sample[bestAtrribute] for sample in data]  
    # 将特征对应的值放到一个集合中，使得特征列的数据具有唯一性  
    uniqueValues = set(atrributeValues)  
    # 遍历uniqueVals中的值  
    for value in uniqueValues:  
        # 子标签subLabels为labels删除bestFeat标签后剩余的标签  
        subAttribute = attribute[:]
        # myTree为key为bestFeatLabel时的决策树  
        C45Tree[bestAtrributeLabel][value] = createC45Tree(spiltDataSet(data,bestAtrribute, value), subAttribute)  
    # 返回决策树  
    return C45Tree
    
def evaluationFunction(dataSet):
    "评价函数:计算真正例、假反例、假正例、真反例"
    #存储分类的类别名，属性行的最后一个值为类别名
    classLabel = list(dataSet.columns)[-1]
    classification = []
    #找出类别分类的所有结果，即正负分类的名字
    for i in range(len(dataSet)):
        classification.append(dataSet[classLabel].loc[i])
    #转为集合去除重复
    classifiction = set(classification)
    #转为list，方便选取其中的类别
    classifiction = list(classifiction)
    #用classifiction0和classifiction1 分别存储二分类的两个正负样本的标签，这是'yes'和'no'
    #存储正样本的分类值
    classifiction0 =  classifiction[0]
    #存储负样本的分类值
    classifiction1 =  classifiction[1]
    #找出所有的classfiction0的样本
    dataClass0 = dataSet[dataSet[classLabel]==classifiction0]
    #找出所有的classfiction1的样本
    dataClass1 = dataSet[dataSet[classLabel]==classifiction1]
    dataClass0.index = range(len(dataClass0))
    dataClass1.index = range(len(dataClass1))
    
    '''构造随机训练和测试样本'''
    '''对正样本的随机操作'''
    #增加rand列，用于随机构建样本
    dataClass0['rand'] = 0
    length = len(dataClass0)
    listRand = []
    #给rand列产生0到length的随机函数，保证每个值都不同
    for i in range(length):
        rand = np.random.randint(0,length)
        while rand  in listRand:
            rand = np.random.randint(0,length)
        while rand not in listRand:
            dataClass0.rand.loc[i] = rand
            listRand.append(rand)
    dataClass0.sort_values(['rand'],ascending=True,inplace=True)
    dataClass0.index = range(len(dataClass0))
    del dataClass0['rand']
    
    '''对负样本的随机操作'''
    #增加rand列，用于随机构建样本
    dataClass1['rand'] = 0
    length = len(dataClass1)
    listRand = []

    #给rand列产生0到length的随机函数，保证每个值都不同
    for i in range(length):
        rand = np.random.randint(0,length)
        while rand  in listRand:
            rand = np.random.randint(0,length)
        while rand not in listRand:
            dataClass1.rand.loc[i] = rand
            listRand.append(rand)
    dataClass1.sort_values(['rand'],ascending=True,inplace=True)
    dataClass1.index = range(len(dataClass1))
    del dataClass1['rand']
    
    #构造测试集
    testDataSetClass0 = dataClass0[0:int(len(dataClass0)/5)]
    testDataSetClass1 = dataClass1[0:int(len(dataClass1)/5)]
    testDataSetClass = pd.concat([testDataSetClass0,testDataSetClass1])
    testDataSetClass.index = range(len(testDataSetClass))
    #构造训练集
    trainDataSetClass0 = dataClass0[int(len(dataClass0)/5):]
    trainDataSetClass1 = dataClass1[int(len(dataClass1)/5):]
    trainDataSetClass = pd.concat([trainDataSetClass0,trainDataSetClass1])
    trainDataSetClass.index = range(len(trainDataSetClass))
    #传入训练集，调用构建决策树函数，返回树模型
    treeModel = createC45Tree(trainDataSetClass.values,list(trainDataSetClass.columns)[:-1])
    #存储测试集真实的标签
    testClassTrue = testDataSetClass[classLabel]
    testClassPredict = []
    for i in range(len(testDataSetClass)):
        #存储属性
        attribute = list(trainDataSetClass.columns)[:-1]
        #存储样本属性的值
        testAtrributeData = testDataSetClass.values[i][:-1]
        a = classify(treeModel,attribute,testAtrributeData)
        testClassPredict.append(a)
    #分别计算真正例、假反例、假正例、真反例
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    #循环所有测试集的长度
    for i in range(len(testClassTrue)):
        #如果预测对了正样本：TP加1
        if (testClassTrue[i] == testClassPredict[i]) & (testClassTrue[i] == classifiction0):
            TP += 1
        #如果预测对了负样本：TN加1
        elif (testClassTrue[i] == testClassPredict[i]) & (testClassTrue[i] == classifiction1):
            TN += 1
        #如果预测错了样本分类，且真实样本为正样本：FN加1
        elif (testClassTrue[i] != testClassPredict[i]) & (testClassTrue[i] == classifiction0):
            FN += 1
        #如果预测错了样本分类，且真实样本为负样本：FP加1
        elif (testClassTrue[i] != testClassPredict[i]) & (testClassTrue[i] == classifiction1):
            FP += 1
    #如果当前抽样导致P和R无法计算（分母为0），则返回-1，进行下一次抽样评估
    if ((TP + FP) == 0) | ((TP + FN) == 0):
        return -1,-1,-1,-1
    #返回混淆矩阵：真正例、假反例、假正例、真反例的值
    return TP,FN,FP,TN
    
def repeatedEvaluate(n,dataSet):
    "留出法需要多次随机抽样验证模型"
    #真正例、假反例、假正例、真反例
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    length = n
    #留出法进行多次随机抽样验证模型
    for i in range(length):
        a,b,c,d = evaluationFunction(dataSet)
        print('随机采样训练第'+str(i+1)+'次完成!')
        if a==-1:
            n -= 1
        else:
            TP = TP + a
            FN = FN + b
            FP = FP + c
            TN = TN + d
    TP = TP/n
    FN = FN/n
    FP = FP/n
    TN = TN/n
    #存储查全和查准率
    P = 0
    R = 0
    #计算评估指标
    P = float(TP)/float(TP+FP)
    R = float(TP)/float(TP+FN)
    F1 = float(2*P*R)/(P+R)
    ACC = float(TP+TN)/(TP+TN+FN+FP)
    #设置评估值的精度
    P = float("%.3f" % P)
    R = float("%.3f" % R)
    F1 = float("%.3f" % F1)
    ACC = float("%.3f" % ACC)
    print('P(查准率)='+str(P) + ' R(查全率)='+str(R) + ' F1='+str(F1) + ' ACC(准确率)='+str(ACC))
        
#决策树分类函数  
def classify(inputTree,attribute,testDataSample):
    classLabel = []
    # 得到树中的第一个特征  
    firstStr = list(inputTree.keys())
    firstStr = firstStr[0]
    # 得到第一个对应的值  
    secondDict = inputTree[firstStr]  
    # 得到树中第一个特征对应的索引  
    featIndex = attribute.index(firstStr)  
    # 遍历树  
    for key in secondDict.keys():  
        # 如果在secondDict[key]中找到testVec[featIndex]  
        if testDataSample[featIndex] == key:  
            # 判断secondDict[key]是否为字典  
            if type(secondDict[key]).__name__ == 'dict':  
                # 若为字典，递归的寻找testVec  
                classLabel = classify(secondDict[key], attribute, testDataSample)  
            else:  
                # 若secondDict[key]为标签值，则将secondDict[key]赋给classLabel  
                classLabel = secondDict[key] 
    # 返回类标签  
    return classLabel    
    
    
    
    
if __name__ == '__main__':
#    #测试数据文件导入函数
#    dataAttribute,attribute,dataSet = loadData('weather.nominal.csv')
#    print(data)
#    print(attribute)
#    print(dataSet)
#    
#    #测试计算信息熵
#    #获取除属性的值和属性和数据集
#    dataAttribute,attribute,dataSet = loadData('weather.nominal.csv')
#    #传入数据，返回信息熵
#    comentropy = calculateComentropy(dataAttribute)
#    print(comentropy)
#
#    #测试划分数据集
#    #获取除属性的值和属性
#    dataAttribute,attribute,dataSet = loadData('weather.nominal.csv')
#    #传入数据，返回信息熵
#    spiltedData = spiltDataSet(dataAttribute,1,'cool')
#    print(spiltedData)
#
#    "测试选择最好的用于划分数据集的特征"
#    dataAttribute,attribute,dataSet = loadData('weather.nominal.csv')
#    bestAtrribute = gainBeatAttribute(dataAttribute)
#    print(bestAtrribute)
#
    #随机采样100次，取平均结果，评估函数
    data,attribute,dataSet = loadData('weather.nominal.csv')
    repeatedEvaluate(100,dataSet)

    
    #weather所有样本数据建树并打印出来
    data,attribute,dataSet = loadData('weather.nominal.csv')
    c45Tree = createC45Tree(data,attribute)
    print('weather数据集所有样本数据建树结果（字典格式）：')
    print(c45Tree)
    


