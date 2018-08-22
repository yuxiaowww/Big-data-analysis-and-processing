# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 15:12:33 2017

@author: yuwei
"""

import pandas as pd
import numpy as np
#python画图包      
import matplotlib.pyplot as plt
#导入时间包
import time

def loadData(file):
    "传入数据集"
    #读入没有标签名的数据
    data = pd.read_csv(file,header=None)
    data = np.mat(data)
    #返回数据集以及拆分的数据集
    return data,data[:,0],data[:,1]

def getDistanceSimilarity(data0,data1,n):
    "计算距离相似度，n传入为2时，为欧式距离"
    data0 = pd.DataFrame(data0)
    data1 = pd.DataFrame(data1)
    s = 0
    #循环所有数据
    for i in range(len(data0.loc[0])):
        for j in range(len(data0)):
            #s对所有数据距离度量进行求和
            s = s + pow(data0.loc[j][i]- data1.loc[j][i],n)
    #返回求和开n次根
    return pow(s,float(1)/n)
    
def getRandCentroid(data,k):
    "构建随机质心：对数据集data选择k个点作为初始随机质心"
    data = pd.DataFrame(data)
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
    Centroid = np.mat(Centroid.values)
    return Centroid
    
def repeateCalls(data):
    "多次调用KMeans聚类,通过手肘方法确定最优的K值"
    K = [2,3,4,5,6,7,8]
    E = []
    for i in range(2,9):
        print('正在计算K值为'+str(i)+'误差平方和!')
        centroid,clusterResult = myKMeans(data,i)
        e = clusterResult[:,1].sum()
        E.append(e)
    #绘制肘方法图：即K值关于误差平方和的折线图
    plt.figure(figsize=(6, 6))
    #在图中1的位置里添加子图1
    plt.subplot(111)  
    #绘制折线
    plt.plot(np.array(K), np.array(E))
    plt.savefig('KMeans_findK.png')
    print('K值与误差平方和折线图：')
    plt.show()
    #标记最优聚类簇数的下标
    mark = 0
    #记录斜率
    slope = 0
    for i in range(1,len(E)):
        mark = i
        slope = E[i] - E[i-1]
        if abs(slope) < 5:
            break
    #mark坐标减1，因为对应K值位置需要减1
    mark = mark - 1
    KBest = K[mark]
    return KBest
    
def myKMeans(data,k):
    "构建KMeans函数"
    #调用函数生成随机簇中心
    centroid = getRandCentroid(data,k)
    centroidNew = centroid.copy()
    #length指数据文件的总长度
    length = len(data[:,0])
    a = [0,0]
    clusterResult = []
    #构建length*2的矩阵，存储聚类结果和误差平方和
    for i in range(length):
        clusterResult.append(a)
    clusterResult = pd.DataFrame(clusterResult)
    #对原数据集的备份，增加一列cluster保存聚类的结果
    dataBack = pd.DataFrame(data)
    dataBack['cluster'] = -1

    #簇中心改变的标识
    changed = 0
    while changed==0:
        changed = 1
        #循环所有数据
        for i in range(length):
            minDistance = np.inf
            minMark = np.inf
            #对每个数据循环所有的簇数
            for j in range(k):
                #调用函数计算当前数据与每个簇中心的欧式距离
                distanceNow = getDistanceSimilarity(centroid[j,:],data[i,:],2)
                #如果距离更小，保存最小距离，和当前聚类的簇类别
                if distanceNow < minDistance:
                    minDistance = distanceNow
                    minMark = j
                    dataBack['cluster'].loc[i] = minMark
            #存储聚类的类别
            clusterResult.loc[i][0] = minMark
            clusterResult.loc[i][1] = minDistance**2

        #根据聚类类别，重新构建簇中心
        for ki in range(k):
            KI = dataBack[dataBack['cluster']==ki]
            del KI['cluster']
            KI = np.mat(KI.values)
            centroid[ki,:] = np.mean(KI, axis=0)
            
        #判断簇中心是否还在变化，如果在继续变化，则继续做迭代
        for i in range(len(centroid)):
            for j in range(2):
                if centroid[i,j] != float(centroidNew[i,j]):
                    changed = 0
        centroidNew = centroid.copy()
    return np.mat(centroid),np.mat(clusterResult)
    
def drawer(data,clusterResult,centroid):
    "绘制聚类结果图形"
    #figure函数设置绘图框的大小
    plt.figure(figsize=(6, 6))
    #在图中1的位置里添加子图1
    plt.subplot(111)  
    #scatter绘制散点，按数据第一列、第二列，为散点图的x,y坐标，以及聚类后的簇结果为点的颜色，绘制聚类散点结果图
    plt.scatter(np.array(data[:,0]), np.array(data[:,1]), c=np.array(clusterResult[:,0]),s=100)
    #四个类的聚类中为x，y轴，颜色为红色，形状为x，绘制聚类中心
    plt.scatter(np.array(centroid[:,0]), np.array(centroid[:,1]), c='r',marker='x',s=100)
    #给散点图加标题
    plt.title("My-KMeans:Draw Fig to Cluster")
    #设置X轴标签  
    plt.xlabel('X')  
    #设置Y轴标签  
    plt.ylabel('Y')
    #保存绘图
    plt.savefig('KMeans_cluster.png')
    #显示绘制的散点图
    plt.show()      
    
if __name__ == '__main__':
    data,data0,data1 = loadData('StandardOfZscore.csv')    
#    #测试欧式距离
#    d = getDistanceSimilarity(data0,data1,2)
#    print('欧式距离为：' + str(d))
#    
#    #测试生成随机质心,假设K为5和3
#    centroid = getRandCentroid(data,5)
#    print('K为5时，产生的随机质心为：')
#    print(centroid)
#    centroid = getRandCentroid(data,3)
#    print('K为3时，产生的随机质心为：')
#    print(centroid)
#
#

#    KBest = repeateCalls(data)
#    print('通过肘方法确定最优K值为:'+str(KBest))
#    t0 = time.time()
#    centroid, clusterResult = myKMeans(data,KBest)
#    time_interval = time.time() - t0
#    drawer(data,clusterResult,centroid)
#    print('建模耗时:'+str(time_interval)+'s')

    #测试KMeans函数
    t0 = time.time()
    centroid,clusterResult = myKMeans(data,4)
    time_interval = time.time() - t0
    drawer(data,clusterResult,centroid)
    print('建模耗时:'+str(time_interval)+'s')
