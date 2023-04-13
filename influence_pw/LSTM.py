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
from influence_pw.genericNeuralNet import GenericNeuralNet, variable, variable_with_weight_decay

class LSTM(GenericNeuralNet):
    def __init__(self, **kwargs):
        super(GenericNeuralNet, self).__init__(**kwargs)