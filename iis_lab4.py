import random

import numpy as np
import pandas
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.sparse._data import _data_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import AgglomerativeClustering

mushrooms_orig = pandas.read_csv('d/mushrooms.csv')
# rs = 32
rs = int(random.random()*1000)
mushrooms, mushrooms2 = train_test_split(mushrooms_orig, test_size=0.7, random_state=rs)

features = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing',
            'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
            'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
            'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']

target_features = ['cap-shape', 'cap-surface', 'cap-color']

data_labels = mushrooms.loc[:, target_features]


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

    # Сортируем
    sorted_keys = sorted(values_ranks, key=values_ranks.get, reverse=True)
    sorted_dict = {}
    for key in sorted_keys:
        sorted_dict[key] = values_ranks[key]
    # print(sorted_dict)
    # print(gr)

    # Присваиваем порядок с шагом 1 (опционально)
    for i, key in enumerate(sorted_keys):
        values_ranks[key] = i

    return values_ranks


def marks_to_values_by_label_encoder(feature):
    le = LabelEncoder()
    le.fit(mushrooms[feature])
    return le.transform(mushrooms[feature])


mushrooms1 = pandas.DataFrame()
mushrooms2 = pandas.DataFrame()

for feature in target_features:
    if feature == 'class':
        continue

    t = data_labels[feature].map(marks_to_values(feature))
    mushrooms1[feature] = t

for feature in features:
    if feature == 'class':
        continue
    mushrooms2[feature] = marks_to_values_by_label_encoder(feature)


# Скейлер если нужен (врятли)
# scaler = MinMaxScaler()
# t = scaler.fit_transform(data)
# data = pandas.DataFrame(t, columns=target_features)
# print(data.info())

# clustering1 = AgglomerativeClustering(distance_threshold=None, n_clusters=2).fit(mushrooms1)
# clustering1 = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(mushrooms1)
clustering2 = AgglomerativeClustering(distance_threshold=None, n_clusters=9).fit(mushrooms2)
# clustering2 = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(mushrooms2)


def plot_scatter():
    fl = len(target_features)
    num = 0
    for i in range(0, fl):
        for j in range(i, fl):
            if i == j:
                continue
            num += 1
            subplot = plt.subplot(1, fl, num)
            subplot.scatter(mushrooms1[target_features[i]], mushrooms1[target_features[j]])
            subplot.set_title(target_features[i] + ' ' + target_features[j])


def plot_scatter_3d():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(mushrooms1[target_features[0]], mushrooms1[target_features[1]], mushrooms1[target_features[2]])


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def do_something(mushrooms, clustering):
    data_labels['cluster'] = clustering.labels_
    mushrooms['cluster'] = clustering.labels_

    desired_width = 320
    pandas.set_option('display.max_rows', desired_width)
    gr = mushrooms.groupby(['cluster', target_features[0], target_features[1], target_features[2]])[
        target_features[0]].count()

    print(mushrooms.groupby(['cluster', target_features[0]])['cluster'].count())
    print(mushrooms.groupby(['cluster', target_features[1]])['cluster'].count())
    print(mushrooms.groupby(['cluster', target_features[2]])['cluster'].count())

    # plot_scatter()
    # plot_scatter_3d()
    plot_dendrogram(clustering, truncate_mode="level", p=3)
    plt.show()


do_something(mushrooms2, clustering2)
