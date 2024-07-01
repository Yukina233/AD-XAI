import os
import pickle

from matplotlib import pyplot as plt

path_project = '/home/yukina/Missile_Fault_Detection/project'

path_loss_save = os.path.join(path_project,
                              'GHL_dataset/log/GHL/train_result/std, window=100, step=10, no_tau2_K=7,deepsad_epoch=20,gan_epoch=50,lam1=1,lam2=0.1,tau1=1/loss/3.pkl')

# 从文件中加载字典对象
with open(path_loss_save, 'rb') as file:
    loss_train = pickle.load(file)

# 访问加载后的字典中的ndarray对象
loss_gen = loss_train['loss_gen']
loss_dis = loss_train['loss_dis']
loss_gen_adv = loss_train['loss_gen_adv']
loss_gen_entropy = loss_train['loss_gen_entropy']
loss_gen_mean_ensemble = loss_train['loss_gen_mean_ensemble']

print('last loss_gen:', loss_gen[-1])
print('last loss_dis:', loss_dis[-1])
print('last loss_gen_adv:', loss_gen_adv[-1])
print('last loss_gen_entropy:', loss_gen_entropy[-1])
print('last loss_gen_mean_ensemble:', loss_gen_mean_ensemble[-1])

# 绘制生成器和判别器的损失
plt.figure()
# plt.plot(loss_gen, label='Generator')
plt.plot(loss_dis, label='Discriminator')
plt.plot(loss_gen_adv, label='Generator Adversarial')
# plt.plot(loss_gen_entropy, label='Generator Entropy')
# plt.plot(loss_gen_mean_ensemble, label='loss_gen_mean_ensemble')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘制生成器的对抗损失和熵损失
plt.figure()
plt.plot(loss_gen_entropy, label='Generator Entropy')
# plt.plot(loss_gen_mean_ensemble, label='loss_gen_mean_ensemble')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘制生成器的对抗损失和熵损失
plt.figure()
# plt.plot(loss_gen_entropy, label='Generator Entropy')
plt.plot(loss_gen_mean_ensemble, label='loss_gen_mean_ensemble')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()