import numpy as np
from functools import reduce
from math import log
from math import exp

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # 切分的词条
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])  # 创建一个空的不重复列表
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 取并集,重复元素只取一个值
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 创建一个其中所含元素都为0的向量
    for word in inputSet:  # 遍历每个词条
        if word in vocabList:  # 如果词条存在于词汇表中，则置1
            returnVec[vocabList.index(word)] +=1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    print(type(trainCategory))
    numTrainDocs = len(trainMatrix)  # 计算训练的文档数目
    numWords = len(trainMatrix[0])  # 计算每篇文档的词条数
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 文档属于侮辱类的概率
    p0Num = np.ones(numWords);
    p1Num = np.ones(numWords)  # 创建numpy.zeros数组,
    p0Denom = 2.0;
    p1Denom = 2.0  # 分母初始化为0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:  # 统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])  ## 该词条的总的词数目   这压样求得每个词条出现的概率 P(w1),P(w2), P(w3)...
        else:  # 统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect=np.zeros(numWords)
    p0Vect=np.zeros(numWords)
    # 将所有特征的条件概率化作log形式，便于将乘法运算化为加法（log(abcd)=loga +logb+logc+logd)，同时防止结果过小超出精度被置零
    for x in range(numWords-1):
        p0Vect[x] = (log(p0Num[x]/p0Denom))
        p1Vect[x] = (log(p1Num[x]/p1Denom))
    # p1Vect = p1Num / p1Denom  # 相除
    # p0Vect = p0Num / p0Denom
    return p0Vect, p1Vect, pAbusive  # 返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # p1 = reduce(lambda x,y:x*y, vec2Classify * p1Vec) * pClass1			#对应元素相乘  这里需要好好理解一下,reduce()
    # p0 = reduce(lambda x,y:x*y, vec2Classify * p0Vec) * (1.0 - pClass1)

    p1= sum(vec2Classify*p1Vec)+log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1-pClass1)
    print('logp0:',p0)
    print('logp1:',p1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts,listClasses = loadDataSet()									#创建实验样本
    myVocabList = createVocabList(listOPosts)								#创建词汇表
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))				#将实验样本向量化
    p0V,p1V,pAb = trainNB0(np.array(trainMat),listClasses)		#训练朴素贝叶斯分类器
    testEntry = ['love', 'my', 'dalmation']									#测试样本1
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))	#测试样本向量化
    if classifyNB(thisDoc,p0V,p1V,pAb):
        print(testEntry,'属于侮辱类')										#执行分类并打印分类结果
    else:
        print(testEntry,'属于非侮辱类')										#执行分类并打印分类结果
    testEntry = ['stupid', 'garbage']										#测试样本2

    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))				#测试样本向量化
    if classifyNB(thisDoc,p0V,p1V,pAb):
        print(testEntry,'属于侮辱类')										#执行分类并打印分类结果
    else:
        print(testEntry,'属于非侮辱类')										#执行分类并打印分类结果



if __name__ == '__main__':
    testingNB()
