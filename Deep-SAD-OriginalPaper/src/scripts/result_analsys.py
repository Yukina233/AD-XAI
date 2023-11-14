import os
import json
import numpy as np

# 假设所有的文件夹都在一个名为"data_folders"的目录下
project_path = '/home/yukina/Missile_Fault_Detection/project/Deep-SAD-OriginalPaper/log/'
file_path = project_path + 'cifar10,group'


# 存储文件夹名称和对应test_auc值的列表
auc_values = []

# 遍历file_path目录下的每个文件夹
for folder_name in os.listdir(file_path):
    # 如果文件夹名中包含"ratioNormal=0.1"，则跳过
    if "ratioNormal=0.0" not in folder_name:
        continue

    folder_path = os.path.join(file_path, folder_name)

    # 确保是文件夹
    if os.path.isdir(folder_path):
        result_json_path = os.path.join(folder_path, 'results.json')

        # 确保result.json文件存在
        if os.path.isfile(result_json_path):
            with open(result_json_path, 'r') as file:
                try:
                    result_data = json.load(file)
                    test_auc = result_data.get('test_auc')
                    if test_auc is not None:
                        auc_values.append((folder_name, test_auc))
                except json.JSONDecodeError:
                    print(f"JSON解析错误：{result_json_path}")
                except KeyError:
                    print(f"没有找到test_auc值：{result_json_path}")

# 根据test_auc值排序，取最小的几个
auc_values.sort(key=lambda x: x[1])

# 计算均值和范围
auc_array = np.array([auc for _, auc in auc_values])
mean_auc = np.mean(auc_array)
var_auc = np.var(auc_array)
range_auc = np.ptp(auc_array)  # ptp (peak to peak) function calculates the range

# 假设我们想要最低的5个AUC值
lowest_auc_values = auc_values[:10]

# 打印结果
print(f"所有test_auc的均值：{mean_auc:.4f}")
print(f"所有test_auc的方差：{var_auc:.4f}")
print(f"所有test_auc的范围：{range_auc:.4f}")
print("\n最低的10个test_auc值：")
for folder_name, test_auc in lowest_auc_values:
    print(f"Folder: {folder_name}, Test AUC: {test_auc}")
