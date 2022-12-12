import numpy as np
from tensorflow import keras

num_classes = 10
input_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# x_train 60000张28*28的图片，图片上为0-9的数字   y_train：60000个标签，对应于x_train
# x_test：10000张28*28的图片  y_test：10000个标签，对应于x_test

# 除255是为了归一化
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# np.expand_dims():insert a new axis that will appear at the axis position in the expanded array
# shape（插入一个新轴，该轴将出现在展开的阵列形状中的轴位置）
print('x_shape: ', x_train.shape)  # (60000, 28, 28)
x_train = np.expand_dims(x_train, -1)
print('x_shape: ', x_train.shape)  # (60000, 28, 28)

x_test = np.expand_dims(x_test, -1)
print("训练集大小:", x_train.shape)
print(x_train.shape[0], "训练样本")
print(x_test.shape[0], "测试样本")

print(y_train)

# keras.utils.to_categorical函数是把类别标签转换为onehot编码（categorical就是类别标签的意思，表示现实世界中你分类的各类别），而onehot编码是一种方便计算机处理的二元编码。
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

'''
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

batch_size = 128
epochs = 15 # 训练十二轮

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]) # 编译

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1) # 训练


score = model.evaluate(x_test, y_test, verbose=0)
print("测试 loss:", score[0])
print("测试 accuracy:", score[1])



'''
