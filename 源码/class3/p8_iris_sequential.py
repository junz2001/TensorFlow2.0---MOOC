import tensorflow as tf
from sklearn import datasets
import numpy as np

x_train = datasets.load_iris().data
y_train = datasets.load_iris().target
# 打乱顺序
np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)

# 全连接层网络结构（激活函数为softmax；正则化为l2）
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())
])

# 配置神经网络的训练方法，优化器为sgd；loss损失函数为；metrics评测指标为：y_是数值，y是独热码
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# 执行训练过程，validation_split=从训练集中划分多少比列给测试集；validation_freq每多少次epoch迭代使用测试集验证一次结果
model.fit(x_train, y_train, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)

# 打印和统计
model.summary()
