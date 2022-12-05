import random

import numpy as np
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier, MLPRegressor

mushrooms = pandas.read_csv('d/mushrooms.csv')
rs = int(random.random()*1000)
# rs = 32

features = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing',
            'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
            'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
            'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']

# features = ['cap-color', 'odor', 'gill-size', 'stalk-shape', 'ring-number',  'spore-print-color', 'population']
# features = ['cap-color', 'stalk-shape', 'ring-number', 'population']


def marks_to_values_by_label_encoder(feature):
    le = LabelEncoder()
    le.fit(mushrooms[feature])
    return le.transform(mushrooms[feature])


def score(model):
    y_predicted = model.predict(X_test)
    y_predicted_class = np.zeros(len(y_predicted))
    for i, val in enumerate(y_predicted):
        if val >= 0.5:
            y_predicted_class[i] = 1
        else:
            y_predicted_class[i] = 0
    print('accuracy:', accuracy_score(y_test, y_predicted_class))
    print('precision:', precision_score(y_test, y_predicted_class))
    print('recall:', recall_score(y_test, y_predicted_class))


def select_features(X, y):
    est = RandomForestClassifier()
    selector = RFECV(estimator=est, step=2, min_features_to_select=8)
    selector.fit(X, y)

    X = X.loc[:, selector.support_]
    print(X.info())
    print(selector.ranking_)


le = LabelEncoder()
for feature in features:
    mushrooms[feature] = le.fit_transform(mushrooms[feature])

y = mushrooms['class'].map({'e': 1, 'p': 0})
X = mushrooms.loc[:, features]

# select_features(X, y)

# Скейлер если нужен
scaler = StandardScaler()
# scaler = MinMaxScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=rs)

poisons = 0
for y in y_test:
    if y == 0:
        poisons += 1

print(poisons, len(y_test) - poisons)

# activation{‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default=’relu’
# solver{‘lbfgs’, ‘sgd’, ‘adam’}, default=’adam’
mlpClf = MLPClassifier(hidden_layer_sizes=(12, 12,), activation='relu', solver='adam', max_iter=100, n_iter_no_change=4)
mlpClf.fit(X_train, y_train)

print('Classifier')
score(mlpClf)


# mlpReg = MLPRegressor(hidden_layer_sizes=(100, 100, 100), activation='relu', solver='adam', max_iter=100, n_iter_no_change=5)
# mlpReg.fit(X_train, y_train)
#
# print('Regressor')
# score(mlpClf)
