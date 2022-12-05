import random

import numpy as np
import pandas
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFECV, RFE
from sklearn.feature_selection import SelectKBest, chi2

mushrooms = pandas.read_csv('d/mushrooms.csv')
# rs = 32
rs = int(random.random()*10000)
features = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing',
            'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
            'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
            'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']


# features = ['class', 'cap-color', 'stalk-shape', 'ring-number', 'population']

print(mushrooms.info())
print(mushrooms.head(5))


def marks_to_values_by_label_encoder(feature):
    le = LabelEncoder()
    le.fit(mushrooms[feature])
    return le.transform(mushrooms[feature])


mushrooms2 = pandas.DataFrame(columns=features)

for feature in features:
    if feature == 'class':
        continue
    mushrooms2[feature] = marks_to_values_by_label_encoder(feature)


y = mushrooms[features[0]].map({'e': 1, 'p': 0})
X = mushrooms2.loc[:, features[1:len(features)]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=rs)

clf = DecisionTreeClassifier(max_depth=10, min_samples_leaf=1, criterion='gini', random_state=rs)
# clf.fit(X_train, y_train)

rfecv = RFECV(estimator=clf)
rfe = RFE(estimator=clf, n_features_to_select=6)
pipe = Pipeline([('Feature Selection', rfecv), ('Model', clf)])

pipe.fit(X_train, y_train)

print("Важность признаков: ", rfecv.support_, rfecv.ranking_)
print(pandas.DataFrame(data=rfecv.support_, index=X.columns))

kbest = SelectKBest(chi2, k=9)
kbest.fit(X_train, y_train)

print('Важность признаков: ', kbest.get_support())

pred = pipe.predict(X_test)
print('accuracy: ', accuracy_score(y_test, pred))
print('precision: ', precision_score(y_test, pred))
print('recall:', recall_score(y_test, pred))


def test_data_info():
    poisons = 0
    for y in y_test:
        if y == 0:
            poisons += 1

    print('Характер тестовой выборки: ')
    print('Кол-во ядовитых: ', poisons)
    print('Кол-во съедобных: ', len(y_test) - poisons)


