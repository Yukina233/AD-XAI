import pandas as pd


path_project = '/home/yukina/Missle_Fault_Detection/project/'

# CE_result = pd.read_csv(path_project + 'models/CE_performance.csv')
BCE_result = pd.read_csv(path_project + 'models/BCE_performance_100seeds.csv')

# CE_accuracy = CE_result['accuracy']
# CE_f1 = CE_result['f1']
# CE_recall = CE_result['recall']

BCE_accuracy = BCE_result['accuracy']
BCE_f1 = BCE_result['f1']
BCE_recall = BCE_result['recall']
BCE_loss = BCE_result['loss']

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
# plt.plot(CE_accuracy, label='CE')
plt.plot(BCE_accuracy, label='BCE')
plt.xlabel('Seed')
plt.ylabel('Accuracy')


plt.subplot(2, 2, 2)
# plt.plot(CE_f1, label='CE')
plt.plot(BCE_f1, label='BCE')
plt.xlabel('Seed')
plt.ylabel('F1 Score')


plt.subplot(2, 2, 3)
# plt.plot(CE_recall, label='CE')
plt.plot(BCE_recall, label='BCE')
plt.xlabel('Seed')
plt.ylabel('Recall')


plt.subplot(2, 2, 4)
plt.plot(BCE_loss, label='BCE')
plt.xlabel('Seed')
plt.ylabel('Loss')


plt.savefig(path_project + 'models/BCE_performance_seeds=100.png')

plt.show()
