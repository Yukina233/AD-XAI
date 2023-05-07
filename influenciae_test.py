import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from math import ceil

train_ds, test_ds = tfds.load('cats_vs_dogs', split=['train[:80%]', 'train[80%:]'], as_supervised=True)
train_ds = train_ds.map(lambda x, y: (2. * tf.image.convert_image_dtype(tf.image.resize(x, (224, 224)), dtype=tf.float32) - 1., tf.one_hot(y, depth=2)))
test_ds = test_ds.map(lambda x, y: (2. * tf.image.convert_image_dtype(tf.image.resize(x, (224, 224)), dtype=tf.float32) - 1., tf.one_hot(y, depth=2)))


# 创建一个由1到1000的数字组成的列表
data = list(range(1, 1001))

# 创建一个 TensorFlow 数据集
dataset = tf.data.Dataset.from_tensor_slices(data)
d1 = dataset.batch(10)
d2 = dataset.batch(10)
d3 = dataset.batch(10)
print(d1)
print(d2)
print(d3)
for batch in d1.take(20):
    print(batch.numpy())
for batch in d2.take(20):
    print(batch.numpy())

d4 = dataset.take(1)
print('d4 ')
for data in d4:
    print(data.numpy())

d5 = dataset.take(1)
print('d5 ')
for data in d5:
    print(data.numpy())

import pandas as pd
import matplotlib.pyplot as plt

# 假设你的 DataFrame 如下：
data = {
    'id': [3, 1, 4, 2],
    'score': [70, 85, 90, 75]
}
df = pd.DataFrame(data)

# 按照 'id' 列进行排序
sorted_df = df.sort_values(by='id')

# 绘制横轴为 'id'，纵轴为 'score' 的直方图
plt.bar(sorted_df['id'], sorted_df['score'])
plt.xlabel('ID')
plt.ylabel('Score')
plt.title('Scores by ID')
plt.show()