import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open("testSet.txt")
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    fr.close()
    return dataMat,labelMat


def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def gradAscent(dataMatIn,classLables):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLables).transpose()

    m,n = dataMatrix.shape
    alpha = 0.001
    maxCycles = 500
    weights =  np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = labelMat-h
        weights = weights + alpha * dataMatrix.transpose()* error
    return weights

def stocGradAscent0(dataMatIn,classLables):
    dataMatIn = np.array(dataMatIn)
    m,n = dataMatIn.shape
    weights = np.ones(n)
    alpha = 0.01
    for i in range(m):
        h = sigmoid(np.sum(dataMatIn[i]*weights))
        error = classLables[i]-h
        weights=weights+alpha*error*dataMatIn[i]
    return weights

def stocGradAscent1(dataMatIn,classLables,numIter=150):
    dataMatIn = np.array(dataMatIn)
    m,n = dataMatIn.shape
    weights = np.ones(n)
    
    for i in range(numIter):
        dataIndex = list(range(m))
        for j in range(m):
            alpha = 4/(1.0+i+j)+0.01
            index = int(np.random.uniform(0,len(dataIndex)))
            h = sigmoid(np.sum(dataMatIn[index]*weights))

            error = classLables[index]-h
            weights=weights+alpha*error*dataMatIn[index]
            del(dataIndex[index])

    return weights


def plotBestFit(weights):
    dataMat,labelMat = loadDataSet()
    dataArray = np.array(dataMat)
    n = dataArray.shape[0]

    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []

    for i in range(n):
        if(int(labelMat[i]) == 1):
            xcord1.append(dataArray[i,1])
            ycord1.append(dataArray[i,2])
        else:
            xcord2.append(dataArray[i,1])
            ycord2.append(dataArray[i,2])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')

    x = np.arange(-3.0,3.0,0.1)  
    y = (-weights[0] - weights[1]*x)/weights[2] 

    print("x.shape:",x.shape)  
    print("y.shape:",y.shape)  

    ax.plot(x,y)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

'''
dataMat,labelMat = loadDataSet()
weights = stocGradAscent1(dataMat,labelMat)
plotBestFit(weights)
#print("dataMat,labelMat:",dataMat,labelMat)
'''


def classifyVector(inX,weights):
    prob = sigmoid(np.sum(inX*weights))
    if prob >0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    fr_test = open("horseColicTest.txt")
    fr_train = open("horseColicTraining.txt")
    trainingSet = []
    trainingLabels = []
 

    for line in fr_train.readlines():
        curline = line.strip().split('\t')
        lineArr = []
        for i in range(len(curline)-1):
            lineArr.append(float(curline[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(curline[-1]))
        
    #start training
    #print(trainingLabels)
    weights = stocGradAscent1(trainingSet,trainingLabels)

    errorcnt = 0
    testCnt = 0.0
    for line in fr_test.readlines():
        testCnt+=1
        curline = line.strip().split('\t')
        lineArr = []
        for i in range(len(curline)-1):
            lineArr.append(float(curline[i]))
        label = classifyVector(lineArr,weights)
        if(int(label) != int(curline[-1])):
            errorcnt+=1
    print("error rate: %f" %(errorcnt/testCnt))

    fr_test.close()
    fr_train.close()

colicTest()