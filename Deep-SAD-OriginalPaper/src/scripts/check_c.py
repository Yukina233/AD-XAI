import json
import os
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt

sys.path.append('..')
from DeepSAD import DeepSAD

def reconstruction_C(exp_path):
    # load Deep SAD model (center c, network weights, and possibly autoencoder weights)
    deepSAD = DeepSAD()
    with open(os.path.join(exp_path, 'config.json'), 'r') as file:
        try:
            config_data = json.load(file)
            net_name = config_data.get('net_name')
        except json.JSONDecodeError:
            print(f"JSON解析错误：{exp_path + 'config.json'}")
        except KeyError:
            print(f"没有找到test_auc值：{exp_path + 'config.json'}")
    deepSAD.set_network(net_name)
    deepSAD.load_model(model_path=os.path.join(exp_path,'model.tar'), load_ae=True)
    print('Loading model from %s.' % exp_path + 'model.tar')

    c = torch.Tensor(np.array(deepSAD.c)).unsqueeze(0)
    deepSAD.ae_net.eval()
    reconstruction = deepSAD.ae_net.decoder(c).detach().numpy()
    reconstruction_image = reconstruction.squeeze()  # 去掉不必要的维度，如果有的话
    if reconstruction_image.ndim == 3 and reconstruction_image.shape[0] in {1, 3}:
        # 如果是单通道图像，去掉通道维度；如果是三通道，将通道移到最后
        reconstruction_image = np.transpose(reconstruction_image, (1, 2, 0) if reconstruction_image.shape[0] == 3 else (1, 0))

    # 输出图像
    plt.imshow(reconstruction_image, cmap='gray' if reconstruction_image.ndim == 2 else None)
    plt.axis('off')  # 不显示坐标轴
    plt.savefig(exp_path + '/Reconstruction C',bbox_inches='tight')

    # 读取原有的数据
    with open(os.path.join(exp_path, 'ae_results.json'), 'r') as file:
        data = json.load(file)

    # 在字典中添加新的键值对
    data['c'] = deepSAD.c

    # 将新的数据写回文件
    with open(os.path.join(exp_path, 'ae_results.json'), 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    print('c = ', c)

if __name__ == '__main__':
    project_path = '/home/yukina/Missile_Fault_Detection/project/Deep-SAD-OriginalPaper/log/'
    base_path = project_path + 'cifar10'
    for folder_name in os.listdir(base_path):
        # if "ratioNormal=0.0" not in folder_name:
        #     continue

        exp_path = os.path.join(base_path, folder_name)
        if os.path.isdir(exp_path):
            reconstruction_C(exp_path)
