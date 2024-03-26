from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import abc
import sys

import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing, cluster
import scipy.linalg as slin
import scipy.sparse.linalg as sparselin
import scipy.sparse as sparse
from scipy.special import expit

import os.path
import time
import tensorflow as tf
import math

from influence_pw.hessians import hessians
from influence_pw.newGenericNeuralNet import GenericNeuralNet, variable, variable_with_weight_decay

class LSTM(GenericNeuralNet):
    def __init__(self, model_path, **kwargs):
        super().__init__(**kwargs)
        self.model = tf.keras.models.load_model(model_path)

    def get_influence_on_test_loss(self, test_indices, train_idx,
        approx_type='cg', approx_params=None, force_refresh=True, test_description=None,
        loss_type='normal_loss',
        ignore_training_error=False,
        ignore_hessian=False
        ):

        test_grad_loss_no_reg_val = self.get_test_grad_loss_no_reg_val(test_indices, loss_type=loss_type)

        print('Norm of test gradient: %s' % np.linalg.norm(np.concatenate(test_grad_loss_no_reg_val)))

        start_time = time.time()

        if test_description is None:
            test_description = test_indices

        approx_filename = os.path.join(self.train_dir, '%s-%s-%s-test-%s.npz' % (self.model_name, approx_type, loss_type, test_description))
        if ignore_hessian == False:
            if os.path.exists(approx_filename) and force_refresh == False:
                inverse_hvp = list(np.load(approx_filename)['inverse_hvp'])
                print('Loaded inverse HVP from %s' % approx_filename)
            else:
                inverse_hvp = self.get_inverse_hvp(
                    test_grad_loss_no_reg_val,
                    approx_type,
                    approx_params)
                np.savez(approx_filename, inverse_hvp=inverse_hvp)
                print('Saved inverse HVP to %s' % approx_filename)
        else:
            inverse_hvp = test_grad_loss_no_reg_val

        duration = time.time() - start_time
        print('Inverse HVP took %s sec' % duration)


        start_time = time.time()

        num_to_remove = len(train_idx)
        predicted_loss_diffs = np.zeros([num_to_remove])
        for counter, idx_to_remove in enumerate(train_idx):

            if ignore_training_error == False:
                single_train_feed_dict = self.fill_feed_dict_with_one_ex(self.data_sets.train, idx_to_remove)
                train_grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=single_train_feed_dict)
            else:
                train_grad_loss_val = [-(self.data_sets.train.cluster_labels[idx_to_remove] * 2 - 1) * self.data_sets.train.x[idx_to_remove, :]]
            predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp), np.concatenate(train_grad_loss_val)) / self.num_train_examples

        duration = time.time() - start_time
        print('Multiplying by %s train examples took %s sec' % (num_to_remove, duration))

        return predicted_loss_diffs