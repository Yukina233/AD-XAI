from keras.models import Sequential, load_model
import numpy as np

fea = np.load('data/cla_fea.npy')
one_hot_label = np.load('data/one_hot_label.npy')

models_num = 1
models_list = []
predictions_list = []
error_samples_list = []


for i in range(0, models_num):
    model = load_model(f'models/lstm-fcn-{i}.h5')
    models_list.append(model)

    predictions = model.predict(fea)
    predictions_list.append(predictions)

    error_samples = np.argmax(predictions, axis=1) != np.argmax(one_hot_label, axis=1)
    error_samples_list.append(error_samples)

all_wrong_samples = error_samples_list[0]
for i in range(0, models_num):
    all_wrong_samples = all_wrong_samples & error_samples_list[i]

all_wrong_samples_id = np.where(all_wrong_samples == True)[0]

print(f'all wrong samples_id: {all_wrong_samples_id}')

print('First 5 wrong samples:')
for i in range(0, all_wrong_samples_id.shape[0]):
    print(
        f'{all_wrong_samples_id[i]}: True label: {np.argmax(one_hot_label[all_wrong_samples_id[i]])}')
    print('Predicted label:')
    for j in range(0, models_num):
        print(f'{np.argmax(predictions_list[j][all_wrong_samples_id[i]])}')