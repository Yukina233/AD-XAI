import pandas as pd


path_project = '/home/yukina/Missle_Fault_Detection/project/'

CE_result = pd.read_csv(path_project + 'models/CE_performance.csv')
BCE_result = pd.read_csv(path_project + 'models/BCE_performance.csv')

CE_accuracy = CE_result['accuracy']
CE_f1 = CE_result['f1']
CE_recall = CE_result['recall']

BCE_accuracy = BCE_result['accuracy']
BCE_f1 = BCE_result['f1']
BCE_recall = BCE_result['recall']

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.plot(CE_accuracy, label='CE')
plt.plot(BCE_accuracy, label='BCE')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(CE_f1, label='CE')
plt.plot(BCE_f1, label='BCE')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(CE_recall, label='CE')
plt.plot(BCE_recall, label='BCE')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()

plt.savefig(path_project + 'models/CE_BCE_performance.png')

plt.show()
