# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 09:58:03 2017

@author: yuwei
"""

import pandas as pd
import math
import operator
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


def loadData(fileName):
    "获取数据集"
    data = pd.read_csv(fileName,encoding="gb2312")
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
    #定义标签元字典，key为标签，value为标签的数目  
    classificationCount = {}  
    #遍历所有标签  
    for vote in classificationLabel:  
        #如果标签不在元字典对应的key中  
        if vote not in classificationCount.keys():  
            #将标签放到字典中作为key，并将值赋为0  
            classificationCount[vote] = 0  
        #对应标签的数目加1  
        classificationCount[vote] += 1  
    #对所有标签按数目排序  
    sortedClassificationCount = sorted(classificationCount.iteritems(),key=operator.itemgetter(1),reverse=True)  
    #返回数目最多的标签  
    return sortedClassificationCount[0][0]
    
def createC45Tree(data,attribute,attributeStore):
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
    #找到最好的属性  
    bestAtrributeLabel = attribute[bestAtrribute] 
    
    #存储当前属性的长度
    length = len(attribute)
    #建立字典存储每层的属性        
    if length not in attributeStore.keys():
        attributeStore[length] = []
    attributeStore[length].append(bestAtrributeLabel)

    #定义决策树，key为bestFeatLabel，value为空  
    C45Tree = {bestAtrributeLabel:{}}  
    #删除atrribute[bestAtrribute]
    del(attribute[bestAtrribute]) 

    #取出data中bestFeat列的所有值  
    atrributeValues = [sample[bestAtrribute] for sample in data]  
    #将特征对应的值放到一个集合中，使得特征列的数据具有唯一性  
    uniqueValues = set(atrributeValues)
    #遍历uniqueVals中的值  
    for value in uniqueValues:  
        #子标签subLabels为labels删除bestFeat标签后剩余的标签  
        subAttribute = attribute[:]
        #myTree为key为bestFeatLabel时的决策树 
        C45Tree[bestAtrributeLabel][value] = createC45Tree(spiltDataSet(data,bestAtrribute, value), subAttribute,attributeStore)[0]
        #print(C45Tree[bestAtrributeLabel][value])
    return C45Tree,attributeStore
    
def selectAttribute(attributeStore,n):
    "选择前N层的特征"
    selectAttribute = []
    #把n之前所有层的属性都存储下来
    for n in attributeStore.keys():
        selectAttribute.extend(attributeStore[n])
        n += 1
    #转为列表返回
    selectAttribute = list(set(selectAttribute))
    return selectAttribute
    
def evalution5cross(dataSet,attribute):
    "五折交叉验证"
    "对数据转化字符型为数值型"
    #循环所有列
    for i in range(len(dataSet.columns)):
        #将当前列的值转为集合
        dSetDict = {}
        dSet = set(dataSet[dataSet.columns[i]])
        m = 0
        #将集合转为字典，字典的值就是对应的数值型：0,1,2,3...
        for dSetSample in dSet:
            dSetDict[dSetSample] = m
            m += 1
        #再对当前列根据字典为其赋值
        for j in range(len(dataSet)):
            dataSet[dataSet.columns[i]].loc[j] = dSetDict[dataSet[dataSet.columns[i]].loc[j]]

    print('\n')
    print('未进行特征选择结果：')
    #数据部分
    data = dataSet[dataSet.columns[:-1]]
    #标签
    label = list(dataSet[dataSet.columns[-1]])
    #调用分类器模型
    #clf = RandomForestClassifier()
    clf = GaussianNB()
    #调用交叉验证函数
    F1 = cross_validation.cross_val_score(clf,data,label,scoring='f1_macro').mean()
    print('F1='+str(F1))
    
    print('进行特征选择结果：')
    #数据部分，选择特征选择后的结果
    data = dataSet[attribute]
    #标签
    label = list(dataSet[dataSet.columns[-1]])
    #调用分类器模型
    #随机森林
    #clf = RandomForestClassifier()
    clf = GaussianNB() 
    F1 = cross_validation.cross_val_score(clf,data,label,scoring='f1_macro').mean()
    print('F1='+str(F1))
    
if __name__ == '__main__':
    #获取数据集
    data,attribute,dataSet = loadData('weather.nominal.csv')
    #attributeStore存储J48特征选择后每层的特征结果
    attributeStore = {}
    #调用J48建树函数,返回特征选择结果
    c45Tree,attributeStore = createC45Tree(data,attribute,attributeStore)
    print('特征选择每层的特征：')
    print(attributeStore)
    print('\n')
    print('weather数据集所有样本数据建树结果（字典格式）：')
    print(c45Tree)
    #调用特征选择函数，选取前N层特征
    selectAttribute = selectAttribute(attributeStore,3)
    print('\n')
    print('特征选择J48树前3层的结果：')
    print(selectAttribute)
    print('\n')
    evalution5cross(dataSet,selectAttribute)


