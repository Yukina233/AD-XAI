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