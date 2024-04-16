
import os
import pickle


from matplotlib import pyplot as plt


path_project = '/home/yukina/Missile_Fault_Detection/project'


path_loss_save = os.path.join(path_project, 'adversarial_ensemble_AD/log/train_result/K=2,gan_epoch=50,lam=2,tau=1/loss/4.pkl')

# 从文件中加载字典对象
with open(path_loss_save, 'rb') as file:
    loss_train = pickle.load(file)

# 访问加载后的字典中的ndarray对象
loss_gen = loss_train['loss_gen']
loss_dis = loss_train['loss_dis']
loss_gen_adv = loss_train['loss_gen_adv']
loss_gen_entropy = -loss_train['loss_gen_entropy']

# 绘制生成器和判别器的损失
plt.figure()
# plt.plot(loss_gen, label='Generator')
plt.plot(loss_dis, label='Discriminator')
plt.plot(loss_gen_adv, label='Generator Adversarial')
plt.plot(loss_gen_entropy, label='Generator Entropy')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘制生成器的对抗损失和熵损失
plt.figure()
plt.plot(loss_gen_adv, label='Adversarial')
plt.plot(loss_gen_entropy, label='Entropy')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
