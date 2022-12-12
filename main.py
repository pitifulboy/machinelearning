import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# 文件存放路径
file_path = r'C:\Users\123\Desktop\房价.xlsx'
# 使用pandas 读取excel
df = pd.read_excel(file_path, sheet_name=0)
print(df)

# 将面积和房价数据，转化为Numpy数组以方便进一步的处理
x = np.array(df['房子面积'].tolist())
y = np.array(df['房子价格(万）'].tolist())

print(x)
print(y)

# 标准化
x = (x - x.mean()) / x.std()
# 将原始数据以散点图的形式画出
plt.figure()
plt.scatter(x, y, c="g", s=6)
plt.show()

# 在(-2,2)这个区间上取170个点作为画图的基础
x0 = np.linspace(-2, 2, 170)


# 利用Numpy的函数定义训练并返回多项式回归模型的函数
# deg参数代表着模型参数中的n，亦即模型中多项式的次数
# 返回的模型能够根据输入的x（默认是x0），返回相对应的预测的y
def get_model(deg):
    return lambda input_x=x0: np.polyval(np.polyfit(x, y, deg), input_x)


# 根据参数n、输入的x、y返回相对应的损失
def get_cost(deg, input_x, input_y):
    return 0.5 * ((get_model(deg)(input_x) - input_y) ** 2).sum()


# 定义测试参数集并根据它进行各种实验
test_set = (1, 4, 10)
for d in test_set:
    print(d)
    # 输出相应的损失
    print(get_cost(d, x, y))

# 画出相应的图像
plt.scatter(x, y, c="g", s=20)
for d in test_set:
    plt.plot(x0, get_model(d)(), label="degree = {}".format(d))
    # 将横轴、纵轴的范围分别限制在(-2,4)、(〖10〗^2,1×〖10〗^3)
plt.xlim(-2, 4)
plt.ylim(1e2, 1e3)
# 调用legend方法使曲线对应的label正确显示
plt.legend()
plt.show()
