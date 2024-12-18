import argparse

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import os

from bg_utils import pull_away_loss, one_hot, xavier_init, sample_shuffle_spv, sample_shuffle_uspv, sample_Z, draw_trend
from bg_dataset import load_data, load_data_unbal

path_project = '/home/yukina/Missile_Fault_Detection/project_data'

class OCGAN:
    def __init__(self, args):
        self.en_ae = args.en_ae
        self.n_epochs = args.epochs
        self.dim_input = args.img_shape
        self.latent_dim = args.latent_dim
        self.define_model_params()
        self.build_model()

    def define_model_params(self):
        if self.en_ae == 1:
            self.mb_size = 100
        elif self.en_ae == 2:
            self.mb_size = 70
        else:
            self.mb_size = 70

        self.D_dim = [self.dim_input, int(self.latent_dim * 2), self.latent_dim, 2]
        # self.D_dim = [self.dim_input, int(self.latent_dim * 3), int(self.latent_dim * 2), self.latent_dim, 2]
        # self.G_dim = [self.latent_dim, int(self.latent_dim * 2), int(self.latent_dim * 3), self.dim_input]
        self.G_dim = [self.latent_dim, int(self.latent_dim * 2), self.dim_input]
        self.Z_dim = self.G_dim[0]

    def build_model(self):
        self.X_oc = tf.placeholder(tf.float32, shape=[None, self.dim_input])
        self.Z = tf.placeholder(tf.float32, shape=[None, self.Z_dim])
        self.X_tar = tf.placeholder(tf.float32, shape=[None, self.dim_input])
        self.y_tar = tf.placeholder(tf.int32, shape=[None, self.D_dim[-1]])

        # Define Discriminator
        self.D_W1 = tf.Variable(xavier_init([self.D_dim[0], self.D_dim[1]]))
        self.D_b1 = tf.Variable(tf.zeros(shape=[self.D_dim[1]]))
        self.D_W2 = tf.Variable(xavier_init([self.D_dim[1], self.D_dim[2]]))
        self.D_b2 = tf.Variable(tf.zeros(shape=[self.D_dim[2]]))
        self.D_W3 = tf.Variable(xavier_init([self.D_dim[2], self.D_dim[3]]))
        self.D_b3 = tf.Variable(tf.zeros(shape=[self.D_dim[3]]))
        self.theta_D = [self.D_W1, self.D_W2, self.D_W3, self.D_b1, self.D_b2, self.D_b3]
        # self.D_W4 = tf.Variable(xavier_init([self.D_dim[3], self.D_dim[4]]))
        # self.D_b4 = tf.Variable(tf.zeros(shape=[self.D_dim[4]]))
        # self.theta_D = [self.D_W1, self.D_W2, self.D_W3, self.D_W4, self.D_b1, self.D_b2, self.D_b3, self.D_b4]

        # Define Generator
        self.G_W1 = tf.Variable(xavier_init([self.G_dim[0], self.G_dim[1]]))
        self.G_b1 = tf.Variable(tf.zeros(shape=[self.G_dim[1]]))
        self.G_W2 = tf.Variable(xavier_init([self.G_dim[1], self.G_dim[2]]))
        self.G_b2 = tf.Variable(tf.zeros(shape=[self.G_dim[2]]))
        self.theta_G = [self.G_W1, self.G_W2, self.G_b1, self.G_b2]
        # self.G_W3 = tf.Variable(xavier_init([self.G_dim[2], self.G_dim[3]]))
        # self.G_b3 = tf.Variable(tf.zeros(shape=[self.G_dim[3]]))
        # self.theta_G = [self.G_W1, self.G_W2, self.G_W3, self.G_b1, self.G_b2, self.G_b3]

        # Define Target Network
        self.T_W1 = tf.Variable(xavier_init([self.D_dim[0], self.D_dim[1]]))
        self.T_b1 = tf.Variable(tf.zeros(shape=[self.D_dim[1]]))
        self.T_W2 = tf.Variable(xavier_init([self.D_dim[1], self.D_dim[2]]))
        self.T_b2 = tf.Variable(tf.zeros(shape=[self.D_dim[2]]))
        self.T_W3 = tf.Variable(xavier_init([self.D_dim[2], self.D_dim[3]]))
        self.T_b3 = tf.Variable(tf.zeros(shape=[self.D_dim[3]]))
        self.theta_T = [self.T_W1, self.T_W2, self.T_W3, self.T_b1, self.T_b2, self.T_b3]
        # self.T_W4 = tf.Variable(xavier_init([self.D_dim[3], self.D_dim[4]]))
        # self.T_b4 = tf.Variable(tf.zeros(shape=[self.D_dim[4]]))
        # self.theta_T = [self.T_W1, self.T_W2, self.T_W3, self.T_W4, self.T_b1, self.T_b2, self.T_b3, self.T_b4]

    def generator(self, z):
        G_h1 = tf.nn.relu(tf.matmul(z, self.G_W1) + self.G_b1)
        G_logit = tf.nn.tanh(tf.matmul(G_h1, self.G_W2) + self.G_b2)
        # G_h2 = tf.nn.relu(tf.matmul(G_h1, self.G_W2) + self.G_b2)
        # G_logit = tf.nn.tanh(tf.matmul(G_h2, self.G_W3) + self.G_b3)
        return G_logit

    def discriminator(self, x):
        D_h1 = tf.nn.relu(tf.matmul(x, self.D_W1) + self.D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, self.D_W2) + self.D_b2)
        D_logit = tf.matmul(D_h2, self.D_W3) + self.D_b3
        # D_h3 = tf.nn.relu(tf.matmul(D_h2, self.D_W3) + self.D_b3)
        # D_logit = tf.matmul(D_h3, self.D_W4) + self.D_b4
        D_prob = tf.nn.softmax(D_logit)
        # return D_prob, D_logit, D_h3
        return D_prob, D_logit, D_h2

    def discriminator_tar(self, x):
        T_h1 = tf.nn.relu(tf.matmul(x, self.T_W1) + self.T_b1)
        T_h2 = tf.nn.relu(tf.matmul(T_h1, self.T_W2) + self.T_b2)
        T_logit = tf.matmul(T_h2, self.T_W3) + self.T_b3
        # T_h3 = tf.nn.relu(tf.matmul(T_h2, self.T_W3) + self.T_b3)
        # T_logit = tf.matmul(T_h3, self.T_W4) + self.T_b4
        T_prob = tf.nn.softmax(T_logit)
        # return T_prob, T_logit, T_h3
        return T_prob, T_logit, T_h2

    # def pretrain_target(self, x_pre, y_pre):
    #     y_tar = tf.placeholder(tf.int32, shape=[None, self.D_dim[3]])
    #     T_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.discriminator(x_pre)[1], labels=y_tar))
    #     T_solver = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(T_loss, var_list=self.theta_T)
    #     with tf.Session() as sess:
    #         sess.run(tf.global_variables_initializer())
    #         sess.run(T_solver, feed_dict={self.X_tar: x_pre, y_tar: y_pre})

    def fit(self, x_train, y_train, x_test, y_test):
        y_real_mb = one_hot(np.zeros(self.mb_size), 2)
        y_fake_mb = one_hot(np.ones(self.mb_size), 2)

        # Define loss functions and optimizers
        D_loss, G_loss, T_loss = self.define_losses(y_real_mb, y_fake_mb)
        D_solver = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(D_loss, var_list=self.theta_D)
        G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=self.theta_G)
        T_solver = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(T_loss, var_list=self.theta_T)


        # Train Model
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(T_solver, feed_dict={self.X_tar: x_train, self.y_tar: one_hot(np.zeros(len(x_train)), 2)})
            for epoch in range(self.n_epochs):
                X_mb_oc = sample_shuffle_uspv(x_train)
                for n_batch in range(len(x_train) // self.mb_size):
                    batch_data = X_mb_oc[n_batch * self.mb_size:(n_batch + 1) * self.mb_size]
                    sess.run(D_solver, feed_dict={self.X_oc: batch_data, self.Z: sample_Z(self.mb_size, self.Z_dim)})
                    sess.run(G_solver, feed_dict={self.Z: sample_Z(self.mb_size, self.Z_dim), self.X_oc: batch_data})

                print(f'Epoch: {epoch}')
                # Evaluate Model
                self.evaluate_model(sess, x_test, y_test)

    def define_losses(self, y_real, y_gen):
        # Discriminator loss
        D_prob_real, D_logit_real, _ = self.discriminator(self.X_oc)
        G_sample = self.generator(self.Z)
        D_prob_gen, D_logit_gen, _ = self.discriminator(G_sample)
        D_prob_tar, D_logit_tar, D_h2_tar = self.discriminator_tar(self.X_tar)
        D_prob_tar_gen, D_logit_tar_gen, D_h2_tar_gen = self.discriminator_tar(G_sample)

        D_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_logit_real, labels=y_real))
        D_loss_gen = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_logit_gen, labels=y_gen))
        ent_real_loss = -tf.reduce_mean(tf.reduce_sum(D_prob_real * tf.log(D_prob_real), axis=1))
        D_loss = D_loss_real + 0.1 * D_loss_gen + 1.85 * ent_real_loss

        # Generator loss
        pt_loss = pull_away_loss(D_h2_tar_gen)
        T_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_logit_tar, labels=self.y_tar))
        tar_thrld = tf.divide(tf.reduce_max(D_prob_tar_gen[:, -1]) +
                              tf.reduce_min(D_prob_tar_gen[:, -1]), 2)
        indicator = tf.sign(
            tf.subtract(D_prob_tar_gen[:, -1],
                        tar_thrld))
        condition = tf.greater(tf.zeros_like(indicator), indicator)
        mask_tar = tf.where(condition, tf.zeros_like(indicator), indicator)
        G_ent_loss = tf.reduce_mean(tf.multiply(tf.log(D_prob_tar_gen[:, -1]), mask_tar))
        fm_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(D_logit_real - D_logit_gen), axis=1)))
        G_loss = pt_loss + G_ent_loss + fm_loss

        return D_loss, G_loss, T_loss

    def evaluate_model(self, sess, x_test, y_test):
        prob, _, _ = sess.run(self.discriminator(self.X_oc), feed_dict={self.X_oc: x_test})
        y_pred = np.argmax(prob, axis=1)
        conf_mat = classification_report(y_test, y_pred, target_names=['benign', 'vandal'], digits=4)
        acc = accuracy_score(y_test, y_pred)
        print(conf_mat)
        print(f"Accuracy: {acc:.4f}")

    def decision_function(self, data):
        """
        预测一批样本的异常分数。

        Parameters:
        - x_samples: numpy array, 待检测样本的特征矩阵

        Returns:
        - anomaly_scores: numpy array, 每个样本的异常分数
        """
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # 计算判别器的输出概率
            D_prob, _, _ = sess.run(self.discriminator(self.X_oc), feed_dict={self.X_oc: data})

            anomaly_scores = D_prob[:, 1]
            return anomaly_scores

    def generate_samples(self, num_samples):
        """
        使用训练好的生成器生成指定数量的样本。

        Parameters:
        - num_samples: 需要生成的样本数量

        Returns:
        - generated_samples: 生成的样本
        """
        # 生成一个形状为 (num_samples, Z_dim) 的噪声矩阵，噪声通常从标准正态分布中采样

        Z_input = tf.cast(sample_Z(num_samples, self.Z_dim), tf.float32)

        # 使用生成器生成样本
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            G_logit = self.generator(Z_input)  # 生成张量
            generated_samples = sess.run(G_logit)  # 使用 session 获取数值

        return generated_samples

if __name__ == '__main__':
    def prepare_data(x_benign, x_vandal, en_ae):
        """
        预处理数据，根据en_ae参数分割为训练集和测试集。

        Parameters:
        - x_benign: 正常数据样本
        - x_vandal: 异常数据样本
        - en_ae: 数据集标识参数，用于选择不同的数据集结构

        Returns:
        - x_train: 训练数据集
        - x_test: 测试数据集
        - y_test: 测试集标签
        """
        if en_ae == 1:
            x_benign = x_benign[:10000]
            x_vandal = x_vandal[:10000]
            x_train = x_benign[:7000]
            x_test = x_benign[-3000:].tolist() + x_vandal[-3000:].tolist()
        else:
            x_train = x_benign[:700]
            x_test = x_benign[-490:].tolist() + x_vandal[-490:].tolist()

        # 转换为 numpy 数组
        x_test = np.array(x_test)
        y_test = np.zeros(len(x_test))

        # 标签赋值，1 表示异常样本（x_vandal）
        if en_ae == 1:
            y_test[3000:] = 1
        else:
            y_test[490:] = 1

        return x_train, x_test, y_test


    def load_data(scaler, en_ae=1, path_project='/home/yukina/Missile_Fault_Detection/project_data'):
        """
        加载并预处理数据，根据 en_ae 选择不同的数据集并进行缩放。

        Parameters:
        - scaler: 用于数据缩放的缩放器 (MinMaxScaler)
        - en_ae: 数据集选择参数
        - path_project: 项目的数据文件路径

        Returns:
        - x_benign: 预处理后的正常数据样本
        - x_vandal: 预处理后的异常数据样本
        """
        if en_ae == 1:
            # 使用 Wiki 数据集
            benign_path = os.path.join(path_project, "OCAN-master/data/wiki/ben_hid_emd_4_50_8_200_r0.npy")
            vandal_path = os.path.join(path_project, "OCAN-master/data/wiki/val_hid_emd_4_50_8_200_r0.npy")
        elif en_ae == 2:
            # 使用信用卡数据集（带自编码）
            benign_path = os.path.join(path_project, "OCAN-master/data/credit_card/ben_hid_repre_r2.npy")
            vandal_path = os.path.join(path_project, "OCAN-master/data/credit_card/van_hid_repre_r2.npy")
        else:
            # 使用信用卡数据集（不带自编码）
            benign_path = os.path.join(path_project, "OCAN-master/data/raw_credit_card/ben_raw_r0.npy")
            vandal_path = os.path.join(path_project, "OCAN-master/data/raw_credit_card/van_raw_r0.npy")

        # 加载数据
        x_benign = np.load(benign_path)
        x_vandal = np.load(vandal_path)

        # 数据缩放
        x_benign = scaler.fit_transform(x_benign)
        x_vandal = scaler.transform(x_vandal)

        return x_benign, x_vandal

    # 数据处理和模型训练
    en_ae = 1
    min_max_scaler = MinMaxScaler()
    x_benign, x_vandal = load_data(min_max_scaler, en_ae)
    x_train, x_test, y_test = prepare_data(x_benign, x_vandal, en_ae)

    parser = argparse.ArgumentParser('Train your OCGAN')
    parser.add_argument('--dataset', type=str, default='dataset_name', help='mnist | cifar10')
    parser.add_argument('--en_ae', type=int, default=1, help='network config id')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='Latent dimension of Gaussian noise input to Generator')
    parser.add_argument('--img_shape', type=int, default=200)

    args = parser.parse_args()
    model = OCGAN(args)
    model.fit(x_train, None, x_test, y_test)
