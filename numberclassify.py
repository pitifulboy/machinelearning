from sklearn import datasets
from sklearn.model_selection import train_test_split

# 加载数据
boston = datasets.load_boston()
x = boston['data']
y = boston['target']
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=37, test_size=0.2)
