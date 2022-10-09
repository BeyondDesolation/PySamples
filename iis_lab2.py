from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import numpy as np

size = 750
X = np.random.uniform(0, 1, (size, 14))
Y = (10 * np.sin(np.pi*X[:, 0]*X[:, 1]) + 20*(X[:, 2] - .5)**2 + 10 * X[:, 3] + 5*X[:, 4]**5 + np.random.normal(0, 1))
X[:, 10:] = X[:, :4] + np.random.normal(0, 0.025, (size, 4))

ridge = Ridge(alpha=1.0)
ridge.fit(X, Y)

rfg = RandomForestRegressor(max_depth=4, min_samples_leaf=1, min_impurity_decrease=0, ccp_alpha=0)
rfg.fit(X, Y)

linear = LinearRegression()
selector = SelectKBest(f_regression)
selector.fit(X, Y)
# estimator = Pipeline([('selector', selector), ('reg', linear)])
# estimator.fit(X, Y)

features = ["x%s" % i for i in range(1, 15)]
ranks = {}


def rank_to_dict(ranks, names):
    ranks = np.abs(ranks)
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(np.array(ranks).reshape(14, 1)).ravel()
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks))


ranks["Ridge"] = rank_to_dict(ridge.coef_, features)
ranks["RFR"] = rank_to_dict(rfg.feature_importances_, features)
ranks["f_regression"] = rank_to_dict(selector.scores_, features)

mean = {}
for model_ranks in ranks.values():
    for key, value in model_ranks.items():
        if key not in mean:
            mean[key] = 0
        mean[key] += value

for key, value in mean.items():
    res = value / len(ranks)
    mean[key] = round(res, 2)

print('Значения')
for r in ranks.items():
    print(r)
print('Среднее')
print(mean)


def sort_features(key_value_list):
    sorted_keys = sorted(key_value_list, key=key_value_list.get, reverse=True)
    sorted_dict = {}
    for key in sorted_keys:
        sorted_dict[key] = key_value_list[key]
    return sorted_dict


print('Отсортированные значения')
for mr in ranks.values():
    print(sort_features(mr))

print('Среднее')
print(sort_features(mean))
