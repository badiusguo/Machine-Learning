from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import operator






# 打开文件,此次应指定编码，


# 读取文件所有内容

    # 打开文件,此次应指定编码，

fr = open("C:/Users/badiu/Documents/GitHub/Machine-Learning/kNN/2.海伦约会/datingTestSet.txt", 'r', encoding='utf-8')
# 读取文件所有内容
arrayOLines = fr.readlines()
# 针对有BOM的UTF-8文本，应该去掉BOM，否则后面会引发错误。
arrayOLines[0] = arrayOLines[0].lstrip('\ufeff')
# 得到文件行数
numberOfLines = len(arrayOLines)
# 返回的NumPy矩阵,解析完成的数据:numberOfLines行,3列
returnMat = np.zeros((numberOfLines, 3))
# 返回的分类标签向量
classLabelVector = []
# 行的索引值
index = 0
inX=[0,0,0]
for line in arrayOLines:
    # s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
    line = line.strip()
    # 使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片。
    listFromLine = line.split('\t')
    # 将数据前三列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
    returnMat[index, :] = listFromLine[0:3]
    # 根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
    # 对于datingTestSet2.txt  最后的标签是已经经过处理的 标签已经改为了1, 2, 3
    if listFromLine[-1] == 'didntLike':
        classLabelVector.append(1)
    elif listFromLine[-1] == 'smallDoses':
        classLabelVector.append(2)
    elif listFromLine[-1] == 'largeDoses':
        classLabelVector.append(3)
    index += 1
print(returnMat)
print(inX-returnMat)
# classLabelVector