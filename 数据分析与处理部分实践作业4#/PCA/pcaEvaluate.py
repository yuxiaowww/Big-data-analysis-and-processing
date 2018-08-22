# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 14:47:44 2017

@author: yuwei
"""

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cross_validation
import time
from sklearn.naive_bayes import GaussianNB

def loadData(filename):
    "获取数据"
    dataSet = pd.read_csv(filename)
    #将数据values转为矩阵形式返回
    data = dataSet[dataSet.columns[:-1]]
    label = dataSet[dataSet.columns[-1]]
    data =  np.mat(data.values)
    #返回原始数据集、特征值、以及标签
    return dataSet,data,label

def GetMean(dataSet):
    "计算均值"
    #存储数据的总和
    sumOfData = 0 
    #存储数据的长度
    lengthOfData = len(dataSet) 
    for i in range(len(dataSet)):
        #循环求和
        sumOfData = sumOfData + float(dataSet[i])
    if lengthOfData != 0 :
        #返回均值
        return sumOfData/lengthOfData
    else:
        return '此数据无均值'
    
def sampleCentralization(data):
    "样本中心化"
    dataMean = []
    #对每列数据调用均值计算函数计算均值
    for i in range(data.shape[1]):
        dMean = GetMean(data[:,i])
        #添加每个属性的均值添加
        dataMean.append(dMean)
    #对样本进行样本中心化
    data = data - dataMean
    return data

def trans(data):
    "求矩阵转置的函数，传入数据格式为列表"
    #构造长度为len(data)空的矩阵
    mat = [[] for i in data[0]]
    #外层循环为data所有的行
    for i in data:
        #内存循环为data一行的长度
        for j in range(len(i)):
            mat[j].append(i[j])
    return mat
    
def getGet(data):
    "计算协方差矩阵"
    #得到中心化矩阵
    data = sampleCentralization(data)
    #获取转置矩阵
    dataT = trans(np.array(data).tolist())
    #将列表转为矩阵
    dataT = np.mat(dataT)
    length = len(data)
    #求协方差矩阵
    covData = (1/length)*(dataT*data)
    return covData

def Jacobi(A):
    "雅克比算法,计算对称矩阵的特征值、特征向量"
    #第一次返回绝对值最大和其在矩阵中的位置
    p,q,maxNum = absMax(A)
    #存储特征向量，初始为E矩阵
    V = np.mat(np.eye(A.shape[0]))
    while maxNum > 0.0001:
        #定义E矩阵，每次循环开始时都要为E矩阵
        Upq = np.mat(np.eye(A.shape[0]))
        #如果两个位置的值相等，直接令为π/4
        if A[p,p] == A[q,q] :
            j = sign(A[p,q]) * math.pi / 4
        #否则计算反余弦值
        else :
            u = 2*A[p,q]/(A[p,p]-A[q,q])
            j = (math.atan(u))/2
        #计算余弦和正弦值，在对应位置填充Upq的值
        Upq[p,p] = math.cos(j)
        Upq[p,q] = -math.sin(j)
        Upq[q,p] = math.sin(j)
        Upq[q,q] = math.cos(j)
        #更新v矩阵，即特征向量矩阵，V*Uoq更新
        V = V*Upq
        #更新A矩阵，即特征值矩阵，Upq.T*A*Upq
        A1 = Upq.T*A*Upq
        A = A1
        #重新计算当前矩阵中的除对角线其余位置绝对值最大值及其位置
        p,q,maxNum = absMax(A)
    #截取对角线作为特征值矩阵
    e = []
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if i==j:
                e.append(A[i,j])
#    e = np.mat(e)
    #返回特征值矩阵和特征向量矩阵
    return e,V
                
def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0
           
def absMax(mat):
    "求一个矩阵绝对值最大值"
    #存储绝对值最大
    a = 0
    #存储绝对值最大的位置
    m = 0; n =0
    for i in range(abs(mat).shape[0]):
        for j in range(abs(mat).shape[1]):
            #计算除主对角线外的绝对值最大值
            if i!=j:
                if a < abs(mat)[i,j]:
                    a = abs(mat)[i,j]
                    #存储当前位置的坐标
                    m=i;n=j
    return m,n,a

def getPca(A_back,k,e,V):
    "PCA降维算法"
    #转特征值list型为array型
    e = np.array(e)
    #对特征值进行降序排序，argsort()返回排序的坐标
    eSort = np.argsort(-e)
    #记录k的取值，即总的特征数目
    K = []
    for i in range(len(e)):
        K.append(i+1)
    #绘制肘方法图：即K值关于特征值方差和的折线图
    plt.figure(figsize=(6, 6))
    #在图中1的位置里添加子图1
    plt.subplot(111)  
    #绘制折线
    plt.plot(np.array(K), np.array(e[eSort]))
    print('K值与特征值方差折线图：')
     #设置X轴标签  
    plt.xlabel('Principal Component Numbers')  
    #设置Y轴标签  
    plt.ylabel('Sum of Variance')
    #保存绘图结果到本地
    plt.savefig('selectK.png')
    plt.show()
    #降序排序，选取前k维的特征向量
    eSort = eSort[0:k]
    VSort = V[:,eSort]
    #降维 
    dataResult = A_back*VSort
    resultMat = (dataResult * VSort.T)
    return dataResult,resultMat

def evalution5cross(dataSet,dataResult,label):
    "交叉验证"
    print('\n')
    print('未进行PCA降维结果：')
    #数据部分
    data = dataSet[dataSet.columns[:-1]]
    #调用分类器模型
    clf = GaussianNB()
    #调用交叉验证函数
    t0 = time.time()
    F1 = cross_validation.cross_val_score(clf,data,label,scoring='f1_weighted').mean()
    t1 = time.time() - t0
    print('建模耗时='+str(t1)+'s')
    print('F1='+str(F1))
    
    print('\n')
    print('进行PCA降维结果：')
    #数据部分
    dataResult = pd.DataFrame(dataResult)
    #调用分类器模型
    clf = GaussianNB()
    #调用交叉验证函数
    t0 = time.time()
    F1 = cross_validation.cross_val_score(clf,dataResult,label,scoring='f1_weighted').mean()
    t1 = time.time() - t0
    print('建模耗时='+str(t1)+'s')
    print('F1='+str(F1))
    
if __name__ == '__main__':
    #获取数据
    dataSet,A,label = loadData('iris.csv')
    #备份数据中心化结果
    A_back = sampleCentralization(A)
    #计算协方差矩阵
    A = getGet(A)
    #雅克比算法计算对称矩阵的特征值和特征向量
    e,V = Jacobi(A)
    #调用PCA算法计算降维结果，并绘制图形
    dataResult,resultMat = getPca(A_back,2,e,V)
    #交叉验证
    evalution5cross(dataSet,resultMat[:,2],label)





