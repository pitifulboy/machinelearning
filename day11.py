import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 获取数据集
dataset = pd.read_csv(r'D:\100-Days-Of-ML\100-Days-Of-ML-Code-master\datasets\Social_Network_Ads.csv')
# 预览数据
# print(dataset)

# →选取特征数据
X = dataset.iloc[:, [2, 3]].values
# print(X)
# →选取结果数据
Y = dataset.iloc[:, 4].values
# print(Y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

