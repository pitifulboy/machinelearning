# P86
from sklearn.datasets import fetch_openml  # 我的sklearn版本为1.0.2
from sklearn.linear_model import SGDClassifier
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# 下载mnist数据集,fetch_openml默认返回的是一个DataFrame，设置as_frame=False返回一个Bunch
# mnist.keys() 可查看所有的键
# data键，包含一个数组，每个实例为一行，每个特征为一列。
# target键，包含一个带有标记的数组
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

# 共有7万张图片，因为图片是28×28像素,所以每张图片有784个特征，每个特征代表了一个像素点的强度，从0（白色）到255（黑色）
x, y = mnist["data"], mnist["target"]  # x.shape=(70000, 784)，y.shape=(70000,)
y = y.astype(np.uint8)  # 注意标签是字符，我们把y转换成整数
# 将数据集分为训练集和测试集
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

# 我们可以看到第一张图片是5
some_digit = x[0]
some_digit_image = some_digit.reshape(28, 28)  # 把长为784的一维数组转换成28x28的二维数组
# imshow用于生成图像，参数cmap用于设置图的Colormap，如果将当前图窗比作一幅简笔画，则cmap就代表颜料盘的配色
plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")  # 关掉坐标轴
plt.show()

print(y[0])

# 使用随机梯度下降（SGD）分类器，比如Scikit-Learn的SGDClassifier类
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
# max_iter最大迭代次数，random_state用于打乱数据，42表示一个随机数种子
sgd_clf = SGDClassifier(max_iter=1000, random_state=42)
sgd_clf.fit(x_train, y_train_5)  # 在整个训练集上进行训练

# 模型预测
print(sgd_clf.predict([some_digit]))  # 返回true
