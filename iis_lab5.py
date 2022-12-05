import random

import numpy as np
import pandas
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler, LabelEncoder

mushrooms = pandas.read_csv('d/mushrooms.csv')
# rs = 32
rs = int(random.random()*1000)
# features = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing',
#             'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
#             'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
#             'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']

# features = ['class', 'cap-color', 'odor', 'gill-size', 'stalk-shape', 'ring-number',  'spore-print-color', 'population']
features = ['class', 'cap-color', 'stalk-shape', 'ring-number', 'population']


def marks_to_values(feature):
    # Группируем по целевому признаку и признаку съедобности
    gr = mushrooms.groupby([feature, features[0]])[features[0]].count()
    edibles = {}
    poisonous = {}
    feature_values = set()
    # Записываем количество съедобных и несъедобных грибов для каждой метки
    for i in gr.index:
        if i[1] == 'e':
            edibles[i[0]] = gr[i]
        elif i[1] == 'p':
            poisonous[i[0]] = gr[i]
        # Фиксируем саму метку
        feature_values.add(i[0])

    values_ranks = {}
    # Путем деления кол-ва съедобных на несъедобных для метки вычисляем ее порядковый ранг
    for key in feature_values:
        values_ranks[key] = edibles.get(key, 0.001) / poisonous.get(key, 0.001)

    sorted_keys = sorted(values_ranks, key=values_ranks.get, reverse=True)
    sorted_dict = {}
    for key in sorted_keys:
        sorted_dict[key] = values_ranks[key]
    print(sorted_dict)
    print(gr)

    return values_ranks


def marks_to_values_by_label_encoder(feature):
    le = LabelEncoder()
    le.fit(mushrooms[feature])
    return le.transform(mushrooms[feature])


mushrooms1 = pandas.DataFrame(columns=features)
mushrooms2 = pandas.DataFrame(columns=features)

for feature in features:
    if feature == 'class':
        continue
    t = mushrooms[feature].map(marks_to_values(feature))
    mushrooms1[feature] = t


for feature in features:
    if feature == 'class':
        continue
    mushrooms2[feature] = marks_to_values_by_label_encoder(feature)


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


# СВОЙ СПОСОБ
y = mushrooms[features[0]].map({'e': 1, 'p': 0})
X = mushrooms1.loc[:, features[1:len(features)]]

# Скейлер если нужен
scaler = StandardScaler()
# scaler = MinMaxScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, stratify=y, random_state=rs)

poisons = 0
for y in y_test:
    if y == 0:
        poisons += 1

print(poisons, len(y_test) - poisons)


# Полиномиальная регрессия
lin_reg = LinearRegression()
polynomial_features = PolynomialFeatures(degree=3)
pipeline = Pipeline([("Linear", polynomial_features), ("linear_regression", lin_reg)])
pipeline.fit(X_train, y_train)
score(pipeline)


# LABEL ENCODER
y = mushrooms[features[0]].map({'e': 1, 'p': 0})
X = mushrooms2.loc[:, features[1:len(features)]]

# Скейлер если нужен
scaler = StandardScaler()
# scaler = MinMaxScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, stratify=y, random_state=rs)

poisons = 0
for y in y_test:
    if y == 0:
        poisons += 1

print(poisons, len(y_test) - poisons)


# Полиномиальная регрессия
lin_reg = LinearRegression()
polynomial_features = PolynomialFeatures(degree=3)
pipeline = Pipeline([("Linear", polynomial_features), ("linear_regression", lin_reg)])
pipeline.fit(X_train, y_train)
score(pipeline)





