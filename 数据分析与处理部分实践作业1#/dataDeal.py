# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 18:39:40 2017

@author: yuwei
"""

import pandas as pd
from pandas import DataFrame
import numpy as np
import math

def LoadData(dataSet):
    data = pd.read_csv(dataSet)
    data.replace(to_replace='NaN', value=0, regex=True, inplace=True)
    return data

def GetMean(dataSet):
    "计算均值"
    sumOfData = 0 #存储数据的总和
    lengthOfData = len(dataSet) #存储数据的长度
    for i in range(len(dataSet)):
        #循环求和
        sumOfData = sumOfData + float(dataSet.loc[i])
    if lengthOfData != 0 :
        #返回均值
        return sumOfData/lengthOfData
    else:
        return '此数据无均值'

def GetVar(dataSet):
    "计算方差"
    average = GetMean(dataSet) #得到均值
    lengthOfData = len(dataSet)
    variance = 0
    for i in range(len(dataSet)):
        #首先计算各值与均值差值的平方和
        variance = variance + pow(float(dataSet.loc[i]) - average,2)
    if lengthOfData != 0:
        return variance/lengthOfData
    else:
        return '无标准差'
    
def GetQuantile(dataSet,percent):
    "计算分位数"
    if percent >=1 or percent < 0:
        return '分位数数值传入出错,需要在0-1之间'
    dataSet.sort()
    print(dataSet)
    #如果不为整数，向上取整
    i = math.ceil(len(dataSet)*percent)
    print(i)
    if i > 1 :
        quantile = float(dataSet[i-1])
    else :
        quantile = float(dataSet[0])
    return quantile

def FilterNoiseThreshold(dataSet,maxThreshold,minThreshold):
    "利用最大最小的阈值来过滤噪声数据"
    for i in range(len(dataSet)):
        if float(dataSet.loc[i]) < minThreshold or float(dataSet.loc[i]) > maxThreshold:
            dataSet.loc[i] = np.nan
    return dataSet
    
def GetFillMissing(dataSet):
    "均值和分位数填充缺失值"
    dataSet.replace(to_replace='NaN', value=0, regex=True, inplace=True)
    #均值填充GetMean(dataSet.H3MeK4_N)
    dataSet.H3MeK4_N = dataSet.H3MeK4_N.map(lambda x : GetMean(dataSet.H3MeK4_N) if x==0 else x)
    #分位数填充GetQuantile(dataSet,percent)
    #dataSet.H3MeK4_N = dataSet.H3MeK4_N.map(lambda x : GetQuantile(list(dataSet.H3MeK4_N),0.5) if x==0 else x)
    return dataSet
    
def GetFillMissingHeuristic(data):
    "启发式补全"
    sex = data.sex
    height = data.height
    data.replace(to_replace='NaN', value=-1, regex=True, inplace=True)
    sumHeight0 = 0
    len0 = 0
    sumHeight1 = 0
    len1 = 0
    for i in range(len(data)):
        if (int(sex.loc[i]) == 0) & (int(height.loc[i])!=-1):
            sumHeight0 += height.loc[i]
            len0 += 1
        elif (int(sex.loc[i]) == 1) & (int(height.loc[i])!=-1):
            sumHeight1 += height.loc[i]
            len1 += 1
    for i in range(len(data)):
        if (int(data['height'].loc[i]) == -1) & (int(data['sex'].loc[i]) == 0):
           data['height'].loc[i] = sumHeight0/len0
        if (int(data['height'].loc[i]) == -1) & (int(data['sex'].loc[i]) == 1):
           data['height'].loc[i] = sumHeight1/len1
    return data
    
def GetWidthSpiltbox(data,width):
    "等宽装箱：每个箱子用均值填充"
    data.sort() #首先进行排序
    print(data)
    t=0
    i=width
    box = []
    for line in data:
        #判断当前值是否大于第一个值加宽度，如果大于就需要放进下一个箱子
        if line >= (data[t]+i): 
            #返回line这个值在data列表中的位置
            ind = data.index(line)
            #为当前箱子赋值
            box.append(data[t:ind+1])
            #当前游标为索引值再后移1个单位
            t = ind
            t += 1
        #如果line已经到达最后了，不管是否达到宽度都属于最后一个箱子
        elif line == data[len(data)-1]:
            ind = data.index(line)
            box.append(data[t:ind+1])
    print('设定区间范围为:'+str(width))
    for i in range(len(DataFrame(box))):
        print('第'+str(i+1)+'个箱子分箱结果为：')
#        for j in range(len(DataFrame((box[i])))):
#            box[i] = list(DataFrame(box[i]).mean())*len(DataFrame((box[i])))
        print(box[i])
            
def GetFrequencySpiltbox(data,box):
    "等频装箱：每个箱子用均值填充"
    data.sort() #首先进行排序
    print(data)
    flag = len(data) % box #用于判断最后面的箱子是否刚好分完
    dict = {}   #存储于字典中
    #最后的箱子不能分完
    if flag >= 1:
        boxNum = len(data)/box + 1
        i = 0
        j = 0 #盒子长度的移动
        for i in range(int(boxNum)-1):
            dict[i] = data[0+j:box+j]
            j = j + box
#            #均值填充
#            dict[i] = [GetMean(DataFrame(dict[i]))]*len(DataFrame(dict[i]))
        dict[i+1] = data[j:]
        #print(dict[i+1])
        dict[i+1] = [GetMean(DataFrame(dict[i+1]))]*len(DataFrame(dict[i+1]))
    #箱子刚好分完
    else:
        boxNum = len(data)/box
        i = 0
        j = 0
        for i in range(int(boxNum)):
            dict[i] = data[0+j:box+j]
            j = j + box
#            #均值填充
#            dict[i] = [GetMean(DataFrame(dict[i]))]*len(DataFrame(dict[i]))
    print('每个箱子的个数为:'+str(box))
    for value in dict:
        #print('第'+str(value+1)+'个箱子用均值平滑结果')
        print('第'+str(value+1)+'个箱子结果')
        print(dict[value])
    
def GetOneHot(dataSet):
    "oneHot编码"
    i = 0 #做为标记
    length = 0 #one-hot编码长度
    dataSet = list(dataSet) #转化为list
    dataset = set(dataSet) #赋值给dataset
    dataDict = {} #字典来存储数字对应的one-hot编码
    length = len(dataset) #记录集合长度
    oneHotInitial = ['0']*length #存储onehot的初始值000000
    #循环建立one-hot编码的字典
    for key in dataset:
        oneHotInitial[i] = '1'
        i += 1
        oneHotNow = oneHotInitial
        oneHotNow = ''.join(oneHotNow) #将one-hot由列表转为字符串
        dataDict[key] = oneHotNow #对应key值的编码存储到字典中
        oneHotInitial = ['0']*length #再次初始化one-hot编码
    #循环为对应的数值赋予one-hot编码
    for i in range(len(dataSet)):
        dataSet[i] = dataDict[dataSet[i]]
    return dataSet
    
def GetSort(dataSet):
    for i in range(len(dataSet)):
        if dataSet.loc[i] == 'A':
            dataSet.loc[i] = 1
        elif dataSet.loc[i] == 'B':
            dataSet.loc[i] = 2
        elif dataSet.loc[i] == 'C':
            dataSet.loc[i] = 3
        elif dataSet.loc[i] == 'D':
            dataSet.loc[i] = 4
    return dataSet
    
def GetMaxMin(dataSet):
    "最大-最小归一化"
    minNum = dataSet.min()
    maxNum = dataSet.max()
    dataSet.H3MeK4_N = dataSet.H3MeK4_N.map(lambda x : float((x-minNum)/(maxNum-minNum)))
    return dataSet

def GetZscore(dataSet):
    "Z-score归一化"
    average = GetMean(dataSet)
    var = GetVar(dataSet)
    dataSet.H3MeK4_N = dataSet.H3MeK4_N.map(lambda x : float((x-average)/var))
    return dataSet
    
def GetEuclidean(dataSet,n):
    "距离相似度计算"
    s = 0
    for i in range(len(dataSet)):
        s = s + pow(abs(dataSet['H3MeK4_N'].loc[i] - dataSet['CaNA_N'].loc[i]),n)
    return pow(s,float(1)/n)
    
def GetCosine(a,b):
    "余弦相似度计算"
#    s = 0
#    s1 = 0
#    s2 = 0
#    for i in range(len(dataSet)):
#        s = s + (dataSet['H3MeK4_N'].loc[i])*(dataSet['CaNA_N'].loc[i])
#        s1 = s1 + pow(dataSet['H3MeK4_N'].loc[i],2)
#        s2 = s2 + pow(dataSet['CaNA_N'].loc[i],2)
#    return s/((s1**0.5)*(s2**0.5))
    s = 0
    s1 = 0
    s2 = 0
    for i in range(len(a)):
        s = s + (a[i])*(b[i])
        s1 = s1 + pow(a[i],2)
        s2 = s2 + pow(b[i],2)
    return s/((s1**0.5)*(s2**0.5))
    
    
if __name__ == '__main__':
    data = DataFrame((LoadData('dataSet2.csv')['H3MeK4_N']))
    d = DataFrame((LoadData('dataSet2.csv')['H3MeK4_N']))
    a = (LoadData('dataSet1.csv'))
    #data = (LoadData('dataSet2.csv'))
#    #数据的均值
#    average = GetMean(data)
#    print('------均值------')
#    print(average)
#    
#    #数据的方差
#    variance = GetVar(data)
#    print('------方差------')
#    print(variance)
#    
#    #数据的分位数
#    d = [1,5,2,3,7,6,4,8,9,10]
#    quantile = GetQuantile(d,0.3)
#    print('----p分位数----')
#    print(quantile)
#    quantile = GetQuantile(d,0.5)
#    print('----p分位数----')
#    print(quantile)
#  
#    #利用最大最小的阈值来过滤噪声数据
#    filterNoiseData = FilterNoiseThreshold(data,2,0.2)
#    print(filterNoiseData)
#    
#    #等宽装箱
#    widthSpiltbox = GetWidthSpiltbox(data,0.1)
#    
#    a = [1,2,3,4,10,6,7,8,9,5]
#    #等频装箱
#    frequencySpiltbox = GetFrequencySpiltbox(a,5)
#    print('\n')
#    frequencySpiltbox = GetFrequencySpiltbox(a,3)
#    
#    a = [1,2,3,4,10,6,7,8,9,5]
#    #等宽装箱
#    ddd = GetWidthSpiltbox(a,2)
#    print('\n')
#    #等宽装箱
#    ddd = GetWidthSpiltbox(a,3)
#
#
#    d = LoadData('dataSet1.csv')
#    d = pd.get_dummies(d['class'])
#
#    a = [1,2,3,4,5,5,2]
#    #one-hot编码
#    a = DataFrame(a)
#    print(a)
#    onehot = DataFrame(GetOneHot(list(a[0])))
#    print(onehot)
#    
#
#
#    #缺失值的填充
#    fillMissing = GetFillMissing(data)

#    #启发式补全
#    a = pd.read_csv('data1.csv')
#    b = GetFillMissingHeuristic(a)
#
#    #排序编码
#    grade = data['grade']
#    grade_sort = GetSort(grade)
#
#
#    #最大最小归一化
#    maxMin = GetMaxMin(data)
#
#    #Z-score归一化
#    zScore = GetZscore(data)
#
#    #距离相似度
#    euclidean = GetEuclidean(a,2)
#    print('距离相似度为:'+str(euclidean))
#
#    #余弦相似度
#    cosine = GetCosine(a)
#    print('余弦相似度为:'+str(cosine))
#    #余弦相似度
#    a = [1,2,5,4,5]
#    b = [5,5,9,2,5]
#    cosine = GetCosine(a,b)
#    print('余弦相似度为:'+str(cosine))
