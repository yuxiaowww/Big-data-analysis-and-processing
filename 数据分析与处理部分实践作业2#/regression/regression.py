# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:03:16 2017

@author: yuwei
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
    
def loadData(fileName):
    "获取数据文件"
    data = pd.read_csv(fileName)
    #由于计算线性回归函数其中有k和b，其中b导致必须在x后加一列1
    dataX_mat = np.mat(data.loc[:,['x','1']])
    #y就是最后一列y
    dataY_mat = data.y
    return dataX_mat,dataY_mat
    
def trans(data):
    "求矩阵转置的函数，传入数据格式为列表"
    #构造长度为多少列的空矩阵（当前的列就是转置矩阵的行）
    mat = [[] for i in data[0]]
    #外层循环为data所有的行
    for i in data:
        #内存循环为data一行的长度
        for j in range(len(i)):
            #根据下标获取转置后位置，即坐标交换
            mat[j].append(i[j])
    return mat

def rowOperation(data):
    "初等行变换"
    length = len(data)
    #首先添加对应的0矩阵
    for i in range(len(data[0])):
        for j in range(len(data)):
            data[j].extend([0])
    #对角线转化为1，构造单位矩阵
    for i in range(len(data)):
        data[i][i+len(data)] = 1
    '''构建上三角'''
    q1 = 0 #记录列
    q2 = 0 #记录行
    while q1<length:
        a = data[q1][q1]
        #化对角线的值为1
        for i in range(len(data[0])):
            data[q1][i] = float(data[q1][i])/float(a)
        q2 = q1 + 1
        #对当前列，对角线下方的值计算为0
        while q2<length:
            b = data[q2][q1]
            for i in range(2*length):
                data[q2][i] = -data[q1][i]*b + data[q2][i]
            q2 += 1
        q1 += 1
    '''化上三角为E矩阵'''
    q1 = length - 1 #记录行
    q2 = length - 1 #记录列
    while q1>0:
        q2 = q1 - 1
        while q2>-1:
            #对当前列，对角线上方的值计算为0
            b = data[q2][q1]
            #print(b)
            for i in range(2*length):
                data[q2][i] = -data[q1][i]*b + data[q2][i]
            q2 -= 1
        q1 -= 1
    return data
    
def GetI(data):
    "计算矩阵的逆"
    data = rowOperation(data)
    length = len(data)
    #截取矩阵右边行变换结果：矩阵的逆
    data_I = []
    for i in range(length):
        data_I.append(data[i][length:])  
    #设置矩阵的逆中的精度跟调用库函数矩阵的逆的精度一致
    for i in range(len(data_I)):
        for j in range(len(data_I)):
            data_I[i][j] = float(("%.8f" % data_I[i][j]))
    return data_I

    
def inverseOrnot(data):
    "判断矩阵是否可逆，看是不是满秩，等价于行列式的值不为0"
    data = rowOperation(data)
    length = len(data)
    #截取矩阵左边行变换结果：单位矩阵
    data_E = []
    for i in range(length):
        data_E.append(data[i][0:length])
    for i in range(length):
        for j in range(length):
            if data_E == 0:
                return False
    return True
    
def GetanalyticSolutions(x,y):
    "计算解析解"
    '''
    公式：w*(解析解) = (X的转置[带一列1]乘以 X)求逆矩阵 乘以 X的转置[带一列1] 乘以 y
    '''
    dataX_mat = np.mat(x)
    #存储X矩阵的转置乘以X矩阵
    dataX_mat_2 = [] 
    '''调用转置函数，先存储为一行n列的list形式'''
    a = [[]]
    a[0] = list(y)
    #对yMat求转置，即y值应该是一列值
    dataY_mat = trans(a) 
    '''调用转置函数，先转化为array格式'''
    #对xMat求转置，目的是为了求xMat矩阵相乘
    dataX_mat_T = trans(np.array(x))
    #计算x矩阵的逆
    dataX_mat2 = dataX_mat_T*dataX_mat
    '''调用矩阵求逆的函数，需要先把矩阵转为list'''
    #整体的矩阵转为array后才能转为list
    dataX_mat2 = list(np.array(dataX_mat2))
    #分量也需要转为list
    dataX_mat2[0] = list(dataX_mat2[0])
    dataX_mat2[1] = list(dataX_mat2[1])
    dataX_mat_2 = copy.deepcopy(dataX_mat2)
    #判断是否可逆
    if inverseOrnot(dataX_mat2) == False:
        print('该矩阵不是满秩矩阵，不可逆!')
        return
    w = GetI(dataX_mat_2) * (np.mat(dataX_mat_T)*dataY_mat)
    #返回解析解
    return w

def drawImage(x,y,w):
    dataX_mat = np.mat(x)
    #计算预测的y值
    dataY_mat_new = dataX_mat*w
    #绘制画布
    fig = plt.figure()
    #在一行一列的画布里，在一个位置绘制图形
    fig.add_subplot(111)
    #调用trans()函数转置y
    a = [[]]
    a[0] = list(y)
    #对dataY_mat求转置，即y值应该是一列值
    dataY_mat = trans(a)
    "绘制真实值 -> 绘制预测的线性回归直线 -> 保存图片到当前目录下"
    #绘制真实的点，红色的小圆
    plt.plot(dataX_mat[:,0],np.mat(dataY_mat)[:,0],'ro')
    #绘制拟合出的直线，即y值为预测值，黑色的直线来绘制
    plt.plot(dataX_mat[:,0],dataY_mat_new,'k')
    #保存绘制的图到当前目录
    plt.savefig('regression.png')
    #打印出拟合出的直线
    print('拟合出的直线为：'+'y='+str(float(w[0]))+'x+'+str(float(w[1])))
    print('\n')
    print('绘制散点图和拟合直线如下：')
    plt.show()
    
if __name__ == '__main__':
#    #矩阵的转置的测试
#    data1 = [[2,1],
#            [3,4],
#            [5,6]]
#    data2 = [[2,1,3,4],
#            [3,4,2,3],
#            [2,4,5,1],
#            [2,5,6,2]]
#    print('3x2矩阵求转置测试:')
#    print(np.mat(data1));print('\n')
#    print('转置后:')
#    print(np.mat(data1).T)
#    print(np.mat(trans(data1)))
#    print('\n')
#    print('四阶方阵求转置测试:')
#    print(np.mat(data2));print('\n')
#    print('转置后:')
#    print(np.mat(data2).T)
#    print(np.mat(trans(data2)))
#
#    #矩阵的逆的测试
#    data1 = [[2,1],
#            [3,4]]
#    data2 = [[2,1,3,4],
#            [3,4,2,3],
#            [2,4,5,1],
#            [2,5,6,2]]
#    data3 = [[2,1,3,4,5,3],
#            [3,4,2,3,2,4],
#            [2,4,5,2,3,1],
#            [6,5,6,2,2,7],
#            [2,3,1,2,3,1],
#            [5,5,7,4,2,2]]
#    print('二阶矩阵求逆测试:')
#    print(np.mat(data1).I)
#    print(np.mat(GetI(data1)))
#    print('四阶矩阵求逆测试：')
#    print(np.mat(data2).I)
#    print(np.mat(GetI(data2)))
#    print('六阶矩阵求逆测试：')
#    print(np.mat(data3).I)
#    print(np.mat(GetI(data3)))
    
    x,y = loadData('regression.csv')
    w = GetanalyticSolutions(x,y)
    num = input('输入需要预测的x值:')
    #计算预测值
    Y = float(num)*float(w[0])+float(w[1])
    print('线性回归预测值为:'+str(Y))
    print('\n')
    drawImage(x,y,w)
      