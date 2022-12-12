import numpy as np
import operator


# 生成样本数据。数据格式【武打镜头数量，亲吻镜头数量】
def createDataset():
    group = np.array([[1, 101], [5, 89], [2, 100], [6, 80], [8, 99], [108, 5], [90, 1], [70, 10], [118, 20], [115, 8]])
    labels = ['爱情片', '爱情片', '爱情片', '爱情片', '爱情片', '动作片', '动作片', '动作片', '动作片', '动作片']
    return group, labels


def classify(inX, dataSet, labels, k):
    # 获取样本大小
    dataSetSize = dataSet.shape[0]
    # np.tile(a,(2,1))第一个参数为Y轴扩大倍数，第二个为X轴扩大倍数。本例中X轴扩大一倍便为不复制
    # 待检测数据格式调整。
    inX_format = np.tile(inX, (dataSetSize, 1))  # 在列方向上重复dataSetSize次
    '''矩阵计算'''
    # 相减
    diffMat = inX_format - dataSet
    # 平方
    sqDiffMat = diffMat ** 2
    # 求和
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方，计算出距离
    distances = sqDistances ** 0.5
    # argsort()函数：作用是将数组按照从小到大的顺序排序，并返回从小到大的元素对应的索引值
    sortedDistIndices = distances.argsort()

    # 创建空的label标签字典
    classCount = {}

    for i in range(k):
        # sortedDistIndices按照元素从小到大，取元素的索引值生成的。
        # sortedDistIndices[i]，按照从小到大依次获取元素的索引值。
        # labels[sortedDistIndices[i]]，按照从小到大依次获取元素，对应的lable的值。
        voteIlabel = labels[sortedDistIndices[i]]
        # 给label标签字典赋值，按照label标签计数
        # get(voteIlabel, 0) ，表示：没有设置 voteIlabel，输出默认的值  0
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    #
    '''
    sorted(iterable, cmp=None, key=None, reverse=False)
    参数说明：
    iterable -- 可迭代对象。
    cmp -- 比较的函数，这个具有两个参数，参数的值都是从可迭代对象中取出，此函数必须遵守的规则为，大于则返回1，小于则返回-1，等于则返回0。
    key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
    reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）。
    '''
    #  dict.items() 返回的都是视图对象（ view objects）operator.itemgetter(1)按照字典的第二个值排序。reverse = True 降序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


if __name__ == '__main__':
    # 生成样本
    group, labels = createDataset()
    # 设置待分类数据
    test = [101, 20]
    # 进行knn计算。K值设置为3
    test_class = classify(test, group, labels, 3)
    print(test_class)
