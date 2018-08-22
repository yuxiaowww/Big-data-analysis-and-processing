# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 21:11:05 2017

@author: yuwei
"""

import pandas as pd
import numpy as np
#python画图包      
import matplotlib.pyplot as plt

def loadData():
    "获取数据"
    data = pd.read_csv('data.csv')
    return data

def getRandCentroid(data,k):
    "构建随机质心：对数据集data选择k个点作为初始随机质心"
    #增加一列构建随机数
    data['rand'] = 0
    length = len(data)
    listRand = []
    #给rand列产生0到length的随机函数，保证每个值都不同
    for i in range(length):
        rand = np.random.randint(0,length)
        #如果产生随机数有重复，则再次随机生成数
        while rand in listRand:
            rand = np.random.randint(0,length)
        #如果不存在重复，则加入
        while rand not in listRand:
            data.rand.loc[i] = rand
            listRand.append(rand)
    #对数据按rand列升序排序
    data.sort_values(['rand'],ascending=True,inplace=True)
    #重新建立索引
    data.index = range(len(data))
    #删除rand列
    del data['rand']
    #保存随机质心，每次选取前k个数据，保证每次选择为随机点
    Centroid = data[0:k]
    #转为矩阵返回结果
    Centroid = np.mat(Centroid.values,dtype=float)
    return Centroid
    
def distEclud(vec1,vec2,data):
    "计算数据隶属于哪一类的，即概率"
    #对于0类簇中心的欧式距离计算
    o1 = np.linalg.norm(vec1 - data)
    #对于1类簇中心的欧式距离计算
    o2 = np.linalg.norm(vec2 - data)
    #返回函数的隶属度，即概率值
    return o1**2/(o1**2+o2**2),o2**2/(o1**2+o2**2)
   
def getEM(data,centroid,databack):
    data['0'] = 0
    data['1'] = 1
    #拷贝簇中心
    centroidNew = centroid.copy()
    #统计EM迭代的次数
    n = 0
    while True:
        n += 1
        print('进行EM迭代第'+str(n)+'次！')
        
        "E步：计算每个数据对于0、1两类的隶属度,即不断更新属于某类的概率"
        for i in range(len(data)):
                #调用隶属度函数计算属于两类分别的隶属度，即概率
                d1,d2 = distEclud(centroid[0,:],centroid[1,:],databack[i,:])
                #将返回的隶属度反馈给0、1类别
                data['0'].loc[i] = d1
                data['1'].loc[i] = d2
        a00 = 0.0; b00 = 0.0
        a01 = 0.0; b01 = 0.0
        a10 = 0.0; b10 = 0.0
        a11 = 0.0; b11 = 0.0
        #循环整个长度计算新的簇中心
        "公式：X轴的坐标乘以0|1类平方的和/0|1类平方的和，Y轴的坐标乘以0|1类平方的和/0|1类平方的和"
        for i in range(len(data)):
            a00 = a00 + (data['x'].loc[i])*(data['1'].loc[i]**2)
            b00 = b00 + (data['1'].loc[i])**2
            a01 = a01 + (data['y'].loc[i])*(data['1'].loc[i]**2)
            b01 = b01 + (data['1'].loc[i])**2
            a10 = a10 + (data['x'].loc[i])*(data['0'].loc[i]**2)
            b10 = b10 + (data['0'].loc[i])**2
            a11 = a11 + (data['y'].loc[i])*(data['0'].loc[i]**2)
            b11 = b11 + (data['0'].loc[i])**2

        "M步：不断更新簇中心"
        centroid[0,:] = round(a00/b00,2),round(a01/b01,2)
        centroid[1,:] = round(a10/b10,2),round(a11/b11,2)
        #存储标记
        d = 0
        #终止条件，当簇中心更新变化范围很小时结束
        for i in range(len(centroid)):
                    if abs(np.linalg.norm(centroid[i,:] - centroidNew[i,:]))<0.001:
                        d += 1
        #计算更新范围满足阈值的簇中心个数，全部满足则结束循环
        if d == len(centroid):
            break
        centroidNew = centroid.copy()
    print('数据样本以及它们属于0、1类别的隶属度（概率）大小：')
    print(data)
    print('迭代结束的0、1类的簇中心：')
    print(centroid)
    return data
    
def showCate(result):
    "根据迭代的结果的概率值大小值，来聚为两类"
    result['cate'] = -1
    for i in range(len(result)):
        if result['0'].loc[i] > result['1'].loc[i]:
            result['cate'].loc[i] = 0
        else:
            result['cate'].loc[i] = 1
    del result['0'];del result['1']
    print('EM最大期望聚类结果:')
    #绘图
    #figure函数设置绘图框的大小
    plt.figure(figsize=(6, 6))
    #在图中1的位置里添加子图1
    plt.subplot(111) 
    #按数据样本最终可能归属概率更大的样本归类绘图
    plt.scatter(np.array(result['x']), np.array(result['y']), c=np.array(result['cate']),s=100)
    #保存绘图
    plt.savefig('EM2_ProbabilityDistributionModel.png')

if  __name__ == '__main__':
    #获取数据集
    data = loadData()
    #复制两个原始数据待用
    dataSet = data.copy()
    databack = data.copy()
    #转为矩阵，方便计算欧氏距离
    databack = np.mat(databack)
    #调用生成随机质心
    centroid = getRandCentroid(data,2)
    #调用EM期望最大算法
    result = getEM(dataSet,centroid,databack)
    #展示可视化的聚类结果
    showCate(result)
