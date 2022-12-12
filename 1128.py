# 亲和力传播聚类
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import AffinityPropagation
from matplotlib import pyplot

# 生成数据集
"""
make_classification函数
sklearn.datasets.make_classification(n_samples=100, n_features=20, *, n_informative=2, n_redundant=2, n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)

参数	类型	默认值	含义
n_samples	int	100	样本数量
n_features	int	20	特征总数。这些包括n_informative 信息特征、n_redundant冗余特征、 n_repeated重复特征和 n_features-n_informative-n_redundant-n_repeated随机抽取的无用特征。
n_informative	int	2	信息特征的数量。
n_redundant	int	2	冗余特征的数量。这些特征是作为信息特征的随机线性组合生成的。(假设n_informative=F1,F2,…那么n_redundant= aF1+bF2+… a,b,c就是随机数)
n_repeated	int	0	从信息特征和冗余特征中随机抽取的重复特征的数量。
n_classes	int	2	分类问题的类（或标签）数。
n_clusters_per_class	int	2	每个类的集群数。
random_state	int	None	类似随机种子，复现随机数
"""
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,
                           random_state=4)

# 定义模型.Affinity Propagation聚类算法简称AP.
# 2007年Brendan J Frey和Delbert Dueck发表于Science期刊。
# AP算法思想是，网络中所有样本点作为节点，通过节点间传递归属度和吸引度两类信息来计算聚类中心，
# 迭代计算出最优的数个聚类中心，并将剩余节点划分到相应的类中。

'''
AffinityPropagation(damping=0.5, max_iter=200, convergence_iter=15, copy=True, preference=None, affinity=’euclidean’, verbose=False)

函数参数

    damping : float, optional, default: 0.5,阻尼系数,默认值0.5

    max_iter : int, optional, default: 200,最大迭代次数,默认值是200

    convergence_iter : int, optional, default: 15,在停止收敛的估计集群数量上没有变化的迭代次数。默认15

    copy : boolean, optional, default: True,布尔值,可选,默认为true,即允许对输入数据的复制

    preference : array-like, shape (n_samples,) or float, optional,近似数组,每个点的偏好 - 具有较大偏好值的点更可能被选为聚类的中心点。 簇的数量，即集群的数量受输入偏好值的影响。 如果该项未作为参数，则选择输入相似度的中位数作为偏好

    affinity : string, optional, default=``euclidean``目前支持计算预欧几里得距离。 即点之间的负平方欧氏距离。

    verbose : boolean, optional, default: False
'''

model = AffinityPropagation(preference=-600, damping=0.7)
# 匹配模型
model.fit(X)

print(X)
# 为每个示例分配一个集群
yhat = model.predict(X)
# 检索唯一群集
clusters = unique(yhat)
# 为每个群集的样本创建散点图
for cluster in clusters:
    # 获取此群集的示例的行索引
    row_ix = where(yhat == cluster)
    print(cluster)
    print(row_ix)

    # 创建这些样本的散布
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
    # 绘制散点图
pyplot.show()
