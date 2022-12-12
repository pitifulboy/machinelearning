import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# 第1步: 数据预处理。
# 导入数据
dataset = pd.read_csv(r'D:\100-Days-Of-ML\100-Days-Of-ML-Code-master\datasets\50_Startups.csv')
# 将数据与标签拆开
X = dataset.iloc[:, :-1].values
print(X)
Y = dataset.iloc[:, 4].values
print(X)

# 数据数字化
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()


# 躲避虚拟变量陷阱
X = X[:, 1:]
print(X)

# 拆分数据集为训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# 第2步： 在训练集上训练多元线性回归模型
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)




