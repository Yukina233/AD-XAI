import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing, cluster

import scipy.linalg as slin
import scipy.sparse.linalg as sparselin
import scipy.sparse as sparse
from scipy.optimize import fmin_ncg

import os.path
import time
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import array_ops
from keras import backend as K

from influence_pw.hessians import hessian_vector_product
from influence_pw.dataset import DataSet
from tensorflow.keras.losses import BinaryCrossentropy, Reduction
import tensorflow_datasets as tfds


class InfluenceExplainer4Classification:
    """
    This class is used to explain the influence of training examples on the test loss in classification problem.

    ----------
    Parameters:
        model: a tensorflow model
        model_name: a string, the name of the model
        train_ds: a Tensorflow Dataset object, the training dataset
        test_ds: a Tensorflow Dataset object, the test dataset
        output_dir: a string, the directory to save the results
        mini_batch: a boolean, whether to use mini-batch to calculate the Hessian-vector product
        loss_function: a tensorflow loss function, the loss function used to calculate the loss
    """

    def __init__(self, model, model_name, train_ds, test_ds, output_dir, mini_batch=True,
                 loss_function=BinaryCrossentropy(from_logits=False,
                                                  reduction=Reduction.NONE), ** kwargs):
        self.model = model
        self.model_name = model_name
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.loss_function = loss_function
        self.mini_batch = mini_batch
        self.num_train_examples = tf.data.experimental.cardinality(train_ds).numpy()
        self.num_test_examples = tf.data.experimental.cardinality(test_ds).numpy()
        if 'damping' in kwargs:
            self.damping = kwargs.pop('damping')
        else:
            self.damping = 0.0
        self.grad_train_loss = None

    def cal_grad_train_loss(self, loss_function, batch_size):
        if self.grad_train_loss is not None:
            return self.grad_train_loss
        else:
            grad_train_loss = []
            for xs, ys in self.train_ds.batch(batch_size):
                with tf.GradientTape() as tape:
                    tape.watch(self.model.trainable_variables)
                    train_predictions = self.model(xs)
                    train_loss = loss_function(ys, train_predictions)
                temp = tape.gradient(train_loss, self.model.trainable_variables)
                grad_train_loss.append(temp)
            self.grad_train_loss = grad_train_loss

    def get_grad_total_train_loss(self, loss_function, batch_size):
        self.cal_grad_train_loss(loss_function, batch_size)
        return [tf.reduce_sum(grad, axis=0) for grad in zip(*self.grad_train_loss)]

    def minibatch_hessian_vector_val(self, v):

        num_examples = self.num_train_examples
        if self.mini_batch == True:
            batch_size = 100
            assert num_examples % batch_size == 0
        else:
            batch_size = self.num_train_examples

        num_iter = int(num_examples / batch_size)

        hessian_vector_val = None
        for xs, ys in self.train_ds.batch(batch_size):
            hessian_vector_val_temp = hessian_vector_product(ys, xs, v)
            if hessian_vector_val is None:
                hessian_vector_val = [b / float(num_iter) for b in hessian_vector_val_temp]
            else:
                hessian_vector_val = [a + (b / float(num_iter)) for (a, b) in
                                      zip(hessian_vector_val, hessian_vector_val_temp)]

        hessian_vector_val = [a + self.damping * b for (a, b) in zip(hessian_vector_val, v)]

        return hessian_vector_val

    def get_fmin_loss_fn(self, v):

        def get_fmin_loss(x):
            hessian_vector_val = self.minibatch_hessian_vector_val(x)

            return 0.5 * np.dot(np.concatenate(hessian_vector_val), x) - np.dot(np.concatenate(v), x)

        return get_fmin_loss

    def get_fmin_grad_fn(self, v):
        def get_fmin_grad(x):
            hessian_vector_val = self.minibatch_hessian_vector_val(x)

            return np.concatenate(hessian_vector_val) - np.concatenate(v)

        return get_fmin_grad

    def get_fmin_hvp(self, x, p):
        hessian_vector_val = self.minibatch_hessian_vector_val(p)

        return np.concatenate(hessian_vector_val)

    def get_cg_callback(self, v, verbose):
        fmin_loss_fn = self.get_fmin_loss_fn(v)

        def fmin_loss_split(x):
            hessian_vector_val = self.minibatch_hessian_vector_val(x)

            return 0.5 * np.dot(np.concatenate(hessian_vector_val), x), -np.dot(np.concatenate(v), x)

        def cg_callback(x):
            # x is current params
            v = x
            idx_to_remove = 5

            train_grad_loss_val = self.get_grad_total_train_loss(BinaryCrossentropy(reduction=Reduction.NONE), 100)
            predicted_loss_diff = np.dot(np.concatenate(v),
                                         np.concatenate(train_grad_loss_val)) / self.num_train_examples

            if verbose:
                print('Function value: %s' % fmin_loss_fn(x))
                quad, lin = fmin_loss_split(x)
                print('Split function value: %s, %s' % (quad, lin))
                print('Predicted loss diff on train_idx %s: %s' % (idx_to_remove, predicted_loss_diff))

        return cg_callback

    def get_inverse_hvp_cg(self, v, verbose):
        fmin_loss_fn = self.get_fmin_loss_fn(v)
        fmin_grad_fn = self.get_fmin_grad_fn(v)
        cg_callback = self.get_cg_callback(v, verbose)

        fmin_results = fmin_ncg(
            f=fmin_loss_fn,
            x0=np.concatenate(v),
            fprime=fmin_grad_fn,
            fhess_p=self.get_fmin_hvp,
            callback=cg_callback,
            avextol=1e-8,
            maxiter=100)

        return fmin_results

    def get_inverse_hvp(self, v, approx_type='cg', approx_params=None, verbose=True):
        assert approx_type in ['cg', 'lissa']
        if approx_type == 'lissa':
            assert "Not Implemented."
        elif approx_type == 'cg':
            return self.get_inverse_hvp_cg(v, verbose)

    def get_test_grad_loss_no_reg_val(self, test_indices, batch_size=100,
                                      loss_function=BinaryCrossentropy(from_logits=False, reduction=Reduction.NONE)):

        # 一次迭代计算batch_size个测试样本的梯度
        num_iter = int(np.ceil(len(test_indices) / batch_size))

        test_grad_loss_no_reg_val = None
        for i in range(num_iter):
            start = i * batch_size
            end = int(min((i + 1) * batch_size, len(test_indices)))

            with tf.GradientTape() as tape:
                tape.watch(self.model.trainable_variables)
                # 用dataset.skip()方法跳过前面的样本，然后使用dataset.take()方法获取所需的样本，最后使用tfds.as_numpy()方法将dataset转换为numpy数组
                target_samples = self.test_ds.skip(test_indices[start]).take(end - start)
                target_inputs = []
                target_labels = []
                for input, label in tfds.as_numpy(target_samples):
                    target_inputs.append(input)
                    target_labels.append(label)
                target_inputs = np.array(target_inputs)
                target_labels = np.array(target_labels)
                test_predictions = self.model(target_inputs)
                test_loss = loss_function(target_labels,
                                          test_predictions)
            temp = tape.gradient(test_loss, self.model.trainable_variables)

            if test_grad_loss_no_reg_val is None:
                test_grad_loss_no_reg_val = [a * (end - start) for a in temp]
            else:
                test_grad_loss_no_reg_val = [a + b * (end - start) for (a, b) in
                                             zip(test_grad_loss_no_reg_val, temp)]

        test_grad_loss_no_reg_val = [a / len(test_indices) for a in test_grad_loss_no_reg_val]
        return test_grad_loss_no_reg_val

    def get_influence_on_test_loss(self, test_indices, train_idx,
                                   approx_type='cg', approx_params=None, force_refresh=True, test_description=None,
                                   X=None, Y=None):
        # If train_idx is None then use X and Y (phantom points)
        # Need to make sure test_idx stays consistent between models
        # because mini-batching permutes dataset order

        if train_idx is None:
            if (X is None) or (Y is None): raise ValueError('X and Y must be specified if using phantom points.')
            if X.shape[0] != len(Y): raise ValueError('X and Y must have the same length.')
        else:
            if (X is not None) or (Y is not None): raise ValueError(
                'X and Y cannot be specified if train_idx is specified.')

        test_grad_loss_no_reg_val = self.get_test_grad_loss_no_reg_val(test_indices)
        # Calculate the gradient of the test loss w.r.t. the parameters, for the test examples

        print('Norm of test gradient: %s' % np.linalg.norm(np.concatenate(test_grad_loss_no_reg_val)))

        start_time = time.time()

        if test_description is None:
            test_description = test_indices

        approx_filename = os.path.join(self.output_dir,
                                       '%s-%s-%s-test-%s.npz' % (
                                           self.model_name, approx_type, self.loss_function, test_description))
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

        duration = time.time() - start_time
        print('Inverse HVP took %s sec' % duration)

        start_time = time.time()
        if train_idx is None:
            num_to_remove = len(Y)
            predicted_loss_diffs = np.zeros([num_to_remove])
            for counter in np.arange(num_to_remove):
                single_train_feed_dict = self.fill_feed_dict_manual(X[counter, :], [Y[counter]])
                train_grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=single_train_feed_dict)
                predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp),
                                                       np.concatenate(train_grad_loss_val)) / self.num_train_examples

        else:
            num_to_remove = len(train_idx)
            predicted_loss_diffs = np.zeros([num_to_remove])
            for counter, idx_to_remove in enumerate(train_idx):
                single_train_feed_dict = self.fill_feed_dict_with_one_ex(self.data_sets.train, idx_to_remove)
                train_grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=single_train_feed_dict)
                predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp),
                                                       np.concatenate(train_grad_loss_val)) / self.num_train_examples

        duration = time.time() - start_time
        print('Multiplying by %s train examples took %s sec' % (num_to_remove, duration))