from os import listdir
from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

#k-近邻算法
#inX用于需要分类的数据
#dataSet输入训练集
#标签向量labels,
#k表示用于选择最近邻居的数目
def classify0(intX,dataSet,labels,k):
    #shape查看矩阵或者数组的维数，例如(3,4)
    #shape[0]为第二维的长度,就是所谓的行数
    dataSetSize = dataSet.shape[0]
    diffMat = tile(intX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    #按照行累加
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    #argsort()函数将数组的值从小到大排序后，并按照其对应的索引值输出
    #得到每个元素的排序序号
    sortedDistIndicies = distances.argsort()
    #print(sortedDistIndicies)
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]   #排名前k个贴标签
        #print(voteIlabel)
        #get(key,x)从字典中获取key对应的value，没有key的话返回0
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        # line.split('\t') 删除line头部和尾部的空白符
        line = line.strip()
        #将line按照\t分割成一个个的字符（如a\tbb\tcc\td将会被分割成[a,bb,cc,d]
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        #-1表示列表中的最后一列元素，很方便的将最后一列存储到向量classLabelVector中
        labels = {'didntLike' : 1, 'smallDoses' : 2, 'largeDoses' : 3}
        classLabelVector.append(labels[listFromLine[-1]])
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    #每一列的最小值，参数0可以使得函数从列中选取最小值
    minVals = dataSet.min(0)
    #每一列的最大值
    maxVals = dataSet.max(0)
    #计算函数可能的取值范围
    ranges = maxVals - minVals

    #生成与矩阵dataSet相同的全零矩阵
    normDataSet = zeros(shape(dataSet))

    #输出dataSet的行数,即数据条数
    m = dataSet.shape[0]

    #tile(A,2)，在列方向上重复A 2次，默认1行
    #tile(minVals,(m,1)),再行方向上重复minVals m次，列方向上1次
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet, ranges, minVals

def datingClassTest():
    #定义用于测试的数据占总数据的10%
    hoRatio = 0.10
    #调用file2matrix读数据文件，得到矩阵
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    #得到归一化后的矩阵
    normMat, ranges, minVals = autoNorm(datingDataMat)

    #输出normMat矩阵的行数
    m = normMat.shape[0]
    #计算测试向量的数量，决定了normMat向量中哪些数据用于测试，哪些用于分类器的训练样本
    numTestVecs = int(m * hoRatio)
    #错误次数
    errorCount = 0.0
    for i in range(numTestVecs):
        #调用 classify0()将测试数据(总数据的前10%），训练数据(总数据后90%)，训练数据标签，k
        classifierResult = classify0(normMat[i, :],normMat[numTestVecs:m, :], \
                                     datingLabels[numTestVecs:m], 4)
        print("The classifer came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        #若判断结果与测试数据标签不同，则错误次数+1
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("The total error rate is: %f" % (errorCount / float(numTestVecs)))

def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 4)
    print("You will probably like this person: ", resultList[classifierResult - 1])

#手写识别系统
#每个样本都是32行32列=1024大小的二进制图像矩阵转换为1x1024的向量
def img2Vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

#手写识别系统测试代码
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')  #获取训练数据文件名列表，
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  #take off .txt
        classNumStr = int(fileStr.split('_')[0])   #每个样本的标签通过文件名标识，文件名下划线左边的就是该样本实际的数字，所以程序先解析这些数据，把它们存在hwLabels中
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2Vector('trainingDigits/%s' % fileNameStr)  #获取训练样本的属性特征
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2Vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 4)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("\n the total number of errors is: %d" % errorCount)
    print("\n the total error rate is: %f" % (errorCount/float(mTest)))
