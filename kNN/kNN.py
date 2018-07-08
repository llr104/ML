import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt
import os

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = inX-dataSet #np.tile(inX,(dataSetSize,1))-dataSet
    sqrDiffMat = diffMat**2
    sqrDistances = sqrDiffMat.sum(axis=-1)

    #算出距离
    distances = sqrDistances**0.5
    #存放排列的下标
    sortDistancesIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortDistancesIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0)+1
    
    sortClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    lines = fr.readlines()
    numberOfLines = len(lines)
    returnMat = np.zeros((numberOfLines,3))

    labels = []
    index = 0
    for line in lines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,0:3] = listFromLine[:3]
        labels.append(int(listFromLine[-1]))
        index+=1
    return returnMat,labels

def autoNorm(dataSet):
    maxVals = dataSet.max(0)
    minVals = dataSet.min(0)
    ranges = maxVals-minVals

    nornDataSet = np.zeros(dataSet.shape)
    m = dataSet.shape[0]

    nornDataSet = dataSet-np.tile(minVals,(m,1))
    nornDataSet = nornDataSet/np.tile(ranges,(m,1))

    return nornDataSet,minVals,maxVals,ranges


def showData1():
    mat,labels = file2matrix("datingTestSet2.txt")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(mat[:,1],mat[:,2])
    plt.show()

def showData2():
    mat,labels = file2matrix("datingTestSet2.txt")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(mat[:,1],mat[:,2],15.0*np.array(labels),15.0*np.array(labels))
    plt.show()

def showData3():
    mat,labels = file2matrix("datingTestSet2.txt")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(mat[:,0],mat[:,1],15.0*np.array(labels),15.0*np.array(labels))
    plt.show()


def datingClassTest():
    mat,labels = file2matrix("datingTestSet2.txt")
    mat,minVals,maxVals,ranges = autoNorm(mat)
    rate = 0.10
    m = mat.shape[0]
    testNum = int(m*rate)

    errorCount = 0
    for i in range(testNum):
        ret = classify0(mat[i,:],mat[testNum:],labels[testNum:],3)
        if ret != labels[i]:
            errorCount+=1
            print("right is %d,but out put is %d" % (labels[i],ret))

    error = errorCount/testNum*100
    print("error rate: %f" % error)
    print("error count: %d, test count %d" %(errorCount,testNum))

def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    miles = float(input("please input miles:"))
    percent =float(input("please input percent:"))
    ice = float(input("please input ice:"))
    
    mat,labels = file2matrix("datingTestSet2.txt")
    mat,minVals,maxVals,ranges = autoNorm(mat)
    inarray = np.array([miles,percent,ice])

    #print(type(miles))
    #print(inarray)
    inNorm = (inarray-minVals)/ranges

    ret = classify0(inNorm,mat,labels,3)
    print("predict:",resultList[ret-1])

def image2Vector(file):
    f = open(file)
    retvector = np.zeros((1,1024))

    for i in range(32):
        line = f.readline()
        for j in range(32):
            retvector[0,i*32+j] = int(line[j])
    f.close()
    return retvector

def hardwriteClassTest():
    trainFiles = os.listdir("trainingDigits")
    m = len(trainFiles)
    trainData = np.zeros((m,1024))
    labels = []
    index = 0
    for file in trainFiles:
        label = file.split('_')[0]
        trainData[index,:] = image2Vector("trainingDigits/"+file)
        labels.append(label)
        index+=1
    
    errorCount = 0
    testFiles = os.listdir("testDigits")
    testNum = len(testFiles)
    for file in testFiles:
        label = file.split('_')[0]
        testData = image2Vector("testDigits/"+file)
        ret = classify0(testData,trainData,labels,3)
        if(ret != label):
            errorCount+=1
    print("error rate: %f" %(errorCount/testNum))


'''
vector = image2Vector("trainingDigits/0_13.txt")
print(vector[0,0:32])
'''


'''
mat,labels = file2matrix("datingTestSet2.txt")
nornDataSet,minVals,maxVals,ranges = autoNorm(mat)
print(nornDataSet)
'''

'''
showData1()
showData2()
showData3()
'''

'''
group,labels = createDataSet()
#print(group)
#print(labels)
predict = classify0([1.1,0.81],group,labels,3)
print(predict)

mat,labels = file2matrix("datingTestSet2.txt")
print("mat",mat)
print("labels",labels[0:20])
'''

#datingClassTest()
#classifyPerson()

hardwriteClassTest()