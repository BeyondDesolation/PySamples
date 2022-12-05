import matplotlib.pyplot as plt
import pandas
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import hashlib

train = pandas.read_csv('d/train.csv', index_col='PassengerId')
test = pandas.read_csv('d/test.csv', index_col='PassengerId')

features = ['Parch', 'Ticket', 'Fare']
print(train.head(5))
print()
tickets_value_counts = train[features[1]].value_counts(normalize=False)
print('БИЛЕТЫ ', train[features[1]].value_counts())
print('Уникальных значений признака \"билет\":', len(tickets_value_counts))
print()

tickets_values = np.zeros(len(train[features[1]]))
tickets_hash = np.zeros(len(train[features[1]]))
for i, ticket_name in enumerate(train[features[1]]):
    tickets_values[i] = tickets_value_counts[ticket_name]
    tickets_hash[i] = int(hashlib.sha256(ticket_name.encode()).hexdigest(), 16) % 10**8


data_train = train.loc[:, features]
data_train[features[1]] = tickets_values

data_train_hash = data_train.copy()
data_train_hash[features[1]] = tickets_hash

data_train_without_tickets = train.loc[:, [features[0], features[2]]]
y_train = train['Survived']

# Билеты по частотности
print()
print("ДАННЫЕ С TF", data_train)
print()
# создание и обучение дерева решений
clf = DecisionTreeClassifier(random_state=241)
clf.fit(data_train, y_train)
# получение и распечатка важностей признаков
importances = clf.feature_importances_

# Билеты захешированы
print()
print("ДАННЫЕ С ХЕШОМ", data_train_hash)
print()

# создание и обучение дерева решений
clf2 = DecisionTreeClassifier(random_state=241)
clf2.fit(data_train_hash, y_train)
# получение и распечатка важностей признаков
importances2 = clf2.feature_importances_

# Без билетов
clf3 = DecisionTreeClassifier(random_state=241)
clf3.fit(data_train_without_tickets, y_train)
importances3 = clf3.feature_importances_


print(importances)
print(importances2)
print(importances3)
