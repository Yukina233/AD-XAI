from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import copy
import os

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy, Reduction
from tensorflow.keras.models import load_model

from influence_pw.LSTM import LSTM

import influence_pw.dataset as dataset
from influence_pw.dataset import DataSet

# Load the training set and test set
seed = 0
X_train = np.load(f'data_seed={seed}/X_train.npy')
X_test = np.load(f'data_seed={seed}/X_test.npy')
Y_train = np.load(f'data_seed={seed}/Y_train.npy')
Y_test = np.load(f'data_seed={seed}/Y_test.npy')
ID_train = np.load(f'data_seed={seed}/ID_train.npy')
ID_test = np.load(f'data_seed={seed}/ID_test.npy')

train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).map(
    lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.float32)))
test_ds = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).map(
    lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.float32)))



num_classes = 5
batch_size = 500
initial_learning_rate = 0.0001
decay_epochs = [10000, 20000]
keep_probs = [1.0, 1.0]
#
# lstm = LSTM(
#     model_path='models/lstm-fcn-withoutCNN.h5',
#     num_classes=num_classes,
#     batch_size=batch_size,
#     train_sets=train_ds,
#     test_sets=test_ds,
#     initial_learning_rate=initial_learning_rate,
#     keep_probs=keep_probs,
#     decay_epochs=decay_epochs,
#     mini_batch=False,
#     train_dir='pw-output',
#     log_dir='pw-log',
#     model_name='lstm-fcn-withoutCNN')

model = load_model(f'models/lstm-fcn-withoutCNN.h5')

test_idx = 1
train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()

from influence_pw.influenceExplainer import InfluenceExplainer4Classification



influence_explainer = InfluenceExplainer4Classification(
    model=model,
    model_name='lstm-fcn-withoutCNN',
    train_ds=train_ds,
    test_ds=test_ds,
    output_dir='pw-output',
    loss_function=BinaryCrossentropy(from_logits=False,
                                     reduction=Reduction.NONE))
influence_explainer.get_test_grad_loss_no_reg_val(test_indices=[test_idx])
