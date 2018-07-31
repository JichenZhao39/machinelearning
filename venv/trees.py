from math import log

#计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    #返回数据集的行数
    numEntries = len(dataSet)
    #保存每个标签label出现次数的字典
    labelCounts = {}
    #对每组特征向量进行统计
    for featVec in dataSet:
        #提取标签信息
        currentLabel = featVec[-1]
        #如果标签label没有放入统计次数的字典，添加进去
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        #label计数
        labelCounts[currentLabel] += 1
    #香农熵
    shannonEnt = 0.0
    #计算香农熵
    for key in labelCounts:
        #选择该标签的概率
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2) #log base 2
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

#dataSet带划分的数据集
#axis划分数据集的特征
#value特征的返回值
"""
splitDataSet通过遍历dataSet数据集，求出axis对应的column列的值为value的行
就是依据axis列进行分类，如果axis列的数据等于value的时候，就要将axis划分到我们创建的信贷数据集中
"""
def splitDataSet(dataSet, axis, value):
    #创建新的list对象
    retDataSet = []
    for featVec in dataSet:
        #axis列为value的数据集【该数据集需要排除axis列】
        #判断axis列的值是否为value
        if featVec[axis] == value:
            #去掉axis特征
            #[:axis]表示前axis行，即若axis为2，就是去featVec的前axis行
            reducedFeatVec = featVec[: axis]
            #将符合条件的添加到返回的数据集
            #[axis+1:]表示跳过axis的axis+1行，取接下来的数据
            reducedFeatVec.extend(featVec[axis+1:])
            #收集结果值axis列为value的行【该行需要排除axis列】
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    #获得标签个数，（求第一行有多少列的feature，最后一列是label
    numFeatures = len(dataSet[0]) - 1
    #数据集的原始信息熵
    baseEntropy = calcShannonEnt(dataSet)
    #最优的信息增益值，和最优的feature编号
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        #获取对应的feature下的所有数据
        featList = [example[i] for example in dataSet]
        #获取剔重后的集合，使用set对list数据进行去重
        uniqueVals =set(featList)
        #创建一个临时的信息熵
        newEntropy = 0.0
        #遍历某一列的value集合，计算该列的信息熵
        #遍历当前特征中的所有唯一属性值，对每个唯一属性值划分一次数据集，计算数据集的新熵值
        #并对所有唯一特征值得到的熵求和
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            #计算概率
            prob = len(subDataSet) / float(len(dataSet))
            #计算信息熵
            newEntropy += prob * calcShannonEnt(subDataSet)
        #gain[信息增益]：划分数据集前后的信息变化，获取信息熵最大的值
        #信息增益是熵的减少或者是数据无序度的减少，最后，比较所有特征中的信息增益，
        #返回最好特征划分的索引值
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

if __name__ == '__main__':
    myDat, labels = createDataSet()
    print(myDat)
    print(splitDataSet(myDat, 0, 1))
    print(splitDataSet(myDat, 0, 0))
