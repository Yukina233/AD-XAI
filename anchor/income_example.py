path_project = '/home/yukina/Missile_Fault_Detection/project/'

import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from alibi.explainers import AnchorTabular
from alibi.datasets import fetch_adult

seed = 0

adult = fetch_adult()

data = adult.data
target = adult.target
feature_names = adult.feature_names
category_map = adult.category_map

# 输出每个标签所占的比例
label_distr = np.bincount(target.astype('int64')) / len(target)
print('Label distribution:', label_distr)

np.random.seed(seed)
data_perm = np.random.permutation(np.c_[data, target])
data = data_perm[:,:-1]
target = data_perm[:,-1]

idx = 30000
X_train,Y_train = data[:idx,:], target[:idx]
X_test, Y_test = data[idx+1:,:], target[idx+1:]

ordinal_features = [x for x in range(len(feature_names)) if x not in list(category_map.keys())]
ordinal_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                      ('scaler', StandardScaler())])

categorical_features = list(category_map.keys())
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                          ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[('num', ordinal_transformer, ordinal_features),
                                               ('cat', categorical_transformer, categorical_features)])
preprocessor.fit(X_train)

# For debug
transformed_X_train = preprocessor.transform(X_train)

clf = RandomForestClassifier(n_estimators=50)
clf.fit(preprocessor.transform(X_train), Y_train)

predict_fn = lambda x: clf.predict(preprocessor.transform(x))
print('Train accuracy: ', accuracy_score(Y_train, predict_fn(X_train)))
print('Test accuracy: ', accuracy_score(Y_test, predict_fn(X_test)))

explainer = AnchorTabular(predict_fn, feature_names, categorical_names=category_map, seed=1)

explainer.fit(X_train, disc_perc=[25, 50, 75])

idx = 0
class_names = adult.target_names
print('Prediction: ', class_names[explainer.predictor(X_test[idx].reshape(1, -1))[0]])

explanation = explainer.explain(X_test[idx], threshold=0.95)
print('Anchor: %s' % (' AND '.join(explanation.anchor)))
print('Precision: %.2f' % explanation.precision)
print('Coverage: %.2f' % explanation.coverage)


idx = 6
class_names = adult.target_names
print('Prediction: ', class_names[explainer.predictor(X_test[idx].reshape(1, -1))[0]])

explanation = explainer.explain(X_test[idx], threshold=0.95)
print('Anchor: %s' % (' AND '.join(explanation.anchor)))
print('Precision: %.2f' % explanation.precision)
print('Coverage: %.2f' % explanation.coverage)

# 计算所有样本的精度和覆盖率
# X_test = X_test[:5]
explanation_dict = {
            'samle_id': [],
            'label': [],
            'prediction': [],
            'anchor': [],
            'precision': [],
            'coverage': []
        }
preds = explainer.predictor(X_test)
Anchors = []
precisions = []
coverages = []
for i in range (len(X_test)):
    explanation = explainer.explain(X_test[i], threshold=0.95)
    if explanation is not None:
        explanation_dict['samle_id'].append(i)
        explanation_dict['label'].append(Y_test[i])
        explanation_dict['prediction'].append(preds[i])
        explanation_dict['anchor'].append(explanation.anchor)
        explanation_dict['precision'].append(explanation.precision)
        explanation_dict['coverage'].append(explanation.coverage)


print('Average precision: %.2f' % np.mean(explanation_dict['precision']))
print('Average coverage: %.2f' % np.mean(explanation_dict['coverage']))
pickle.dump(explanation_dict, open(path_project + f'anchor/results/label_ratio={round(label_distr[0]/label_distr[1], 2)}.pkl', 'wb'))