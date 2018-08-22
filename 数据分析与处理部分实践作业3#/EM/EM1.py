# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 14:01:44 2017

@author: yuwei
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')


def getData(mu1,mu2,sigma):
    "生成两个随机的正态分布数据集"
    #传入要生成数据集的mu和sigma
    #生成两个正态分布
    data1 = pd.DataFrame(sigma*np.random.randn(1000) + mu1)
    data2 = pd.DataFrame(sigma*np.random.randn(1000) + mu2)
    #用pandas的concat连接两个正态分布数据集
    data = pd.concat([data1,data2])
    #重新建立索引
    data.index = range(len(data))
    return data
    
def getZij(zij):
    "初始时随机生成类别"
    zij[1] = -1
    #用随机函数把原始数据随机分成0、1两类
    for i in range(len(zij)):
        zij[0].loc[i] = np.random.randint(2)
        zij[1].loc[i] = abs(1 - zij[0].loc[i])
    #返回初始化的数据集
    return zij
    
def getEM(data,dataSet,sigma):
    "EM最大期望算法"
    #获取初始随机聚类的数据集
    Zij = getZij(data)
    dataSet['cate'] = 0
    #在原始数据集中增加初始类别
    dataSet['cate'] = Zij[0]
    for i in range(20):
        
        "E的阶段，不断归类"
        #更新两个类别的均值
        #0类别的均值
        u0 = dataSet[dataSet['cate']==0][0].mean()
        #1类别的均值
        u1 = dataSet[dataSet['cate']==1][0].mean()
        #更新两个类别的方差
        #0类别的方差
        sigma0 = (dataSet[dataSet['cate']==0][0].var())**0.5
        #0类别的方差
        sigma1 = (dataSet[dataSet['cate']==1][0].var())**0.5
        for i in range(len(dataSet)):
            if abs(dataSet[0].loc[i]-u0)/sigma0 <= abs(dataSet[0].loc[i]-u1)/sigma1:
                dataSet['cate'].loc[i] = 0
            else:
                dataSet['cate'].loc[i] = 1

        "M阶段，不断进行分类，并设定阈值"
        #如果小于阈值或迭代次数达到20次，结束循环
        if (abs(sigma0-sigma) <= 0.01) & (abs(sigma1-sigma) <= 0.01):
            break
        #重新建立索引
        dataSet.index = range(len(dataSet))
        #分别存储两个类别的数据，返回后进行可视化
        data0 = dataSet[dataSet['cate']==0][0]
        data1 = dataSet[dataSet['cate']==1][0]
    print('\n')
    print('EM期望最大的均值分别为：')
    print('u0='+str(u0)+' u1='+str(u1))
    print('\n')
    return u0,u1,data0,data1
    
if __name__ == '__main__':
    #设置mu和sigma，目的是随机产生两组正态分布的数据
    mu1 = 20
    mu2 = 80
    sigma = 10
    #获取随机生成的正态分布数据
    data = getData(mu1,mu2,sigma)
    #拷贝数据集
    dataBack = data.copy()
    dataSet = data.copy()
    #调用EM算法,返回结果进行可视化
    u0,u1,data0,data1 = getEM(data,dataSet,sigma)
    
    "绘制正态分布参考scipy官网:"
    "https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.normal.html"
    print('EM高斯模型结果:')
    #绘制拟合的正态分布图形
    plt.subplot(211)
    count, bins, ignored = plt.hist(dataBack, 120, normed=True)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - u0)**2 / (2 * sigma**2) ),linewidth=2, color='b')
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - u1)**2 / (2 * sigma**2) ),linewidth=2, color='y')
    #绘制归属于两类的数据
    plt.subplot(212)
    plt.hist(data0,120,alpha=1)
    plt.hist(data1,120,alpha=0.5)
    #保存绘图
    plt.savefig('EM1_GaussianDistributionModel.png')
