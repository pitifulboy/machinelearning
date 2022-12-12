# 第1步：导入库
import numpy as np
import pandas as pd

# 第2步：导入数据集
# 导入数据。可参考pandas官方文档中的read_csv
dataset_df = pd.read_csv(r'D:\100-Days-Of-ML\100-Days-Of-ML-Code-master\datasets\Data.csv')
print(dataset_df)
# 预处理数据。可参考pandas官方文档中的iloc，values函数
# 转换数据（除最后一列外）
X = dataset_df.iloc[:, :-1].values
# print(X)
# 转换数据，最后一列
Y = dataset_df.iloc[:, 3].values
# print(Y)


# 第3步：处理丢失数据
# 将列平均值填充到空值nan中。
from sklearn.impute import SimpleImputer

# 设置计算参数。可参考sklearn官方文档中的SimpleImputer，fit_transform
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
# 计算并替换空值
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])
# print(X[:, 1:3])

# 第4步：解析分类数据。可参考sklearn官方文档中的LabelEncoder, OneHotEncoder，fit_transform
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 将离散值[Country],[Purchased]数据，转换
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# print(X[:, 0])

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
# print(Y)

# 离散型特征进行one-hot编码，使计算距离时更合理。
# One-Hot编码，又称为一位有效编码，主要是采用N位状态寄存器来对N个状态进行编码，每个状态都由他独立的寄存器位，并且在任意时候只有一位有效。
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()
# print(X)

'''测试注释：
test_01 = X[:, 0:1] # X[:, 1:2]  X[:, 2:3]
test_01 = onehotencoder.fit_transform(test_01).toarray()
print(test_01)
'''
# 第5步：拆分数据集为训练集合和测试集合。可参考sklearn官方文档中的train_test_split
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# 第6步：特征标准化。可参考sklearn官方文档中的StandardScaler
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
print(X_train)
X_test = sc_X.transform(X_test)
