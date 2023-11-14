import json
import os
import numpy as np
from collections import defaultdict


def parse_folder_name(folder_name):
    part = folder_name.split(',')
    params = {p.split('=')[0]: p.split('=')[1] for p in part}
    return params


def collect_data_to_groups(base_path):
    groups = defaultdict(list)
    for folder_name in os.listdir(base_path):
        if "ratioNormal=0.0" not in folder_name:
            continue

        if os.path.isdir(os.path.join(base_path, folder_name)):
            params = parse_folder_name(folder_name)
            seed = params.pop('seed')
            key = tuple(sorted(params.items()))  # Create a hashable key from the parameters
            groups[key].append((seed, folder_name))
    print('groups number:', len(groups))
    return groups


def calculate_averages(groups):
    averaged_results = {}
    for group_key, seeds_folders in groups.items():
        auc_sum = 0
        scores_sums = None
        for seed, folder in seeds_folders:
            with open(os.path.join(base_path, folder, 'results.json'), 'r') as file:
                data = json.load(file)
                auc_sum += data['test_auc']
                if scores_sums is None:
                    scores_sums = [0] * len(data['test_scores'])
                for i, score_data in enumerate(data['test_scores']):
                    scores_sums[i] += score_data[2]  # 假设score是列表中的第三个值

        # 平均test_auc
        averaged_auc = auc_sum / len(seeds_folders)

        # 平均test_scores
        averaged_scores = []
        for i, sum_score in enumerate(scores_sums):
            # 假设每个scores的前两个值是标号和标签
            averaged_scores.append(
                [data['test_scores'][i][0], data['test_scores'][i][1], sum_score / len(seeds_folders)])

        # 保存结果
        averaged_results[group_key] = {
            'test_auc': averaged_auc,
            'test_scores': averaged_scores
        }
    return averaged_results


def save_data(averaged_results, save_path):
    for group_key, results in averaged_results.items():
        folder_name = ','.join(f"{k}={v}" for k, v in dict(group_key).items())
        save_folder = os.path.join(save_path, folder_name) + ',seed=avg'
        os.makedirs(save_folder, exist_ok=True)
        with open(os.path.join(save_folder, 'results.json'), 'w') as file:
            json.dump(results, file, indent=4)

        read_folder = os.path.join(base_path, folder_name) + ',seed=0'
        with open(os.path.join(read_folder, 'config.json'), 'r') as file:
            data = json.load(file)
            data['seed'] = -1 #将seed设置为-1，表示平均结果
        with open(os.path.join(save_folder, 'config.json'), 'w') as file:
            json.dump(data, file, indent=4)


base_path = '/home/yukina/Missile_Fault_Detection/project/Deep-SAD-OriginalPaper/log/cifar10'  # Replace with the path to your dataset directories
save_path = '/home/yukina/Missile_Fault_Detection/project/Deep-SAD-OriginalPaper/log/cifar10,group'  # Replace with the path where you want to save the averaged results

# Collect data from all folders
groups = collect_data_to_groups(base_path)

# Calculate averages
averaged_results = calculate_averages(groups)

# Save the averages to files
save_data(averaged_results, save_path)

print("Group result finished!")
