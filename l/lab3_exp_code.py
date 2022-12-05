import random

import pandas
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Это классификация с использованием своего странного метода преобразования меток в значениея.
# Результат не лучше, чем у обычного LabelEncoder, но пусть останентся

rs = int(random.random()*10000)
features = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing',
            'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
            'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
            'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']

mushrooms = pandas.read_csv('../d/mushrooms.csv')

# print(mushrooms.head(5))
# for feature in features:
#     print(mushrooms.groupby([feature, features[0]]).count())


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


mushrooms1 = pandas.DataFrame(columns=features)
for feature in features:
    if feature == 'class':
        continue
    t = mushrooms[feature].map(marks_to_values(feature))
    mushrooms1[feature] = t


# СВОЙ СПОСОБ
y = mushrooms[features[0]].map({'e': 1, 'p': 0})
X = mushrooms1.loc[:, features[1:len(features)]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=rs)

poisons = 0
for y in y_test:
    if y == 0:
        poisons += 1

print(poisons, len(y_test) - poisons)

clf = DecisionTreeClassifier(max_depth=10, min_samples_leaf=1, criterion='gini', random_state=rs)
clf.fit(X_train, y_train)

importances = clf.feature_importances_
print(importances)

pred = clf.predict(X_test)

print("accuracy_score ", accuracy_score(y_test, pred))
print("precision ", precision_score(y_test, pred))
print('recall:', recall_score(y_test, pred))