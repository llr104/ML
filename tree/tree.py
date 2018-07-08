import math
import operator
import matplotlib.pyplot as plt

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts:
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel]+=1
    
    shannonEnt = 0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt-= prob*math.log(prob,2)
    return shannonEnt

def createDataSet():
    dataSet = [ [1,1,'yes'],
                [1,1,'yes'],
                [1,0,'no'],
                [0,1,'no'],
                [0,1,'no']]
    labels = ['aquatic', 'flippers']
    return dataSet,labels

def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if(featVec[axis] == value):
            temp = featVec[:axis]
            temp.extend(featVec[axis+1:])
            retDataSet.append(temp)
    return retDataSet

def chooseBestFeatureToSplit(myData):
    featureNum = len(myData[0])-1
    bestShannonEnt = 0
    bestFeature = -1
    baseShannonEnt = calcShannonEnt(myData)

    for i in range(featureNum):
        featureList = [example[i] for example in myData]
        uniqueList = set(featureList)

        shannonEnt = 0
        for key in uniqueList:
            splitData = splitDataSet(myData,i,key)
            prob = float(len(splitData))/len(myData)
            shannonEnt+=prob*calcShannonEnt(splitData)
        
        infoShannonEnt = baseShannonEnt-shannonEnt
        if(infoShannonEnt>bestShannonEnt):
            bestShannonEnt = infoShannonEnt
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount:
            classCount[vote] = 0
        classCount[vote]+=1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if(classList.count(classList[0])==len(classList)):
       return classList[0]
    
    if(len(dataSet) == 1):
        return majorityCnt(classList)
    
    bestFeature = chooseBestFeatureToSplit(dataSet)
    #print("bestFeature:",bestFeature)
    bestFeatureLabel = labels[bestFeature]
    myTree = {bestFeatureLabel:{}}
    del(labels[bestFeature])

    featureValues = [example[bestFeature] for example in dataSet]
    uniqueList = set(featureValues)
    for value in uniqueList:
        subLabels = labels[:]
        
        subDataSet = splitDataSet(dataSet,bestFeature,value)
        myTree[bestFeatureLabel][value] = createTree(subDataSet,subLabels)

    return myTree


def getNumLeafs(myTree):
    #print("getNumLeafs:",myTree)
    numLeafs = 0

    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        #print(isinstance(secondDict[key],dict))
        if(isinstance(secondDict[key],dict)):
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs+=1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if(isinstance(secondDict[key],dict)):
            thisDepth = getTreeDepth(secondDict[key])+1
        else:
            thisDepth=1
        if(maxDepth<thisDepth):
            maxDepth = thisDepth
    return maxDepth


#在父子节点间填充文本信息
#cntrPt:子节点位置, parentPt：父节点位置, txtString：标注内容
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


#定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8") #定义判断节点形态
leafNode = dict(boxstyle="round4", fc="0.8") #定义叶节点形态
arrow_args = dict(arrowstyle="<-") #定义箭头
 
#绘制带箭头的注解
#nodeTxt：节点的文字标注, centerPt：节点中心位置,
#parentPt：箭头起点位置（上一节点位置）, nodeType：节点属性
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )


#绘制树形图
#myTree：树的字典, parentPt:父节点, nodeTxt：节点的文字标注
def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)  #树叶节点数
    depth = getTreeDepth(myTree)    #树的层数
    firstStr = list(myTree.keys())[0]     #节点标签
    #计算当前节点的位置
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt) #在父子节点间填充文本信息
    plotNode(firstStr, cntrPt, parentPt, decisionNode) #绘制带箭头的注解
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if isinstance(secondDict[key],dict):#判断是不是字典，
            plotTree(secondDict[key],cntrPt,str(key))        #递归绘制树形图
        else:   #如果是叶节点
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
 
#创建绘图区
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    
    plotTree.totalW = float(getNumLeafs(inTree)) #树的宽度
    plotTree.totalD = float(getTreeDepth(inTree)) #树的深度
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0
    plotTree(inTree, (0.5,1.0), '')
    plt.show()

def classify(inputTree,featLabels,testVect):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if(testVect[featIndex] == key):
            if(isinstance(secondDict[key],dict)):
                classLabel = classify(secondDict[key],featLabels,testVect)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree,fileName):
    import pickle
    fw = open(fileName,'wb')
    pickle.dump(inputTree,fw)
    fw.close()

def getTree(fileName):
    import pickle
    fr = open(fileName,"rb")
    return pickle.load(fr)

def createLenses():
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['ages','prescript','astigmatic','tearRate']
    lensesTree = createTree(lenses,lensesLabels)
    print(lensesTree)
    createPlot(lensesTree)

    fr.close()
    return lensesTree

'''
dataSet,labels = createDataSet()
shannonEnt = calcShannonEnt(dataSet)
print("shannonEnt:",shannonEnt)
bestFeature = chooseBestFeatureToSplit(dataSet)
print("bestFeature:",bestFeature)

dataSet = splitDataSet(dataSet,1,1)
print(dataSet)
'''

'''
dataSet,labels = createDataSet()
mytree = createTree(dataSet,labels)
#print(mytree)
#print(getNumLeafs(mytree))
#print(getTreeDepth(mytree))
#createPlot(mytree)
storeTree(mytree,"tree")

label = classify(mytree,labels,[1,1])
print("label:",label)

print(getTree("tree"))
'''
lensesTree = createLenses()
storeTree(lensesTree,"lensesTree")