import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from math import ceil

train_ds, test_ds = tfds.load('cats_vs_dogs', split=['train[:80%]', 'train[80%:]'], as_supervised=True)
train_ds = train_ds.map(lambda x, y: (2. * tf.image.convert_image_dtype(tf.image.resize(x, (224, 224)), dtype=tf.float32) - 1., tf.one_hot(y, depth=2)))
test_ds = test_ds.map(lambda x, y: (2. * tf.image.convert_image_dtype(tf.image.resize(x, (224, 224)), dtype=tf.float32) - 1., tf.one_hot(y, depth=2)))

print('ok')
