from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import numpy as np
from matplotlib import pyplot as plt


size = 750
X = np.random.uniform(0, 1, (size, 14))
Y = (10 * np.sin(np.pi*X[:, 0]*X[:, 1]) + 20*(X[:, 2] - .5)**2 + 10 * X[:, 3] + 5*X[:, 4]**5 + np.random.normal(0, 1))
X[:, 10:] = X[:, :4] + np.random.normal(0, 0.025, (size, 4))

def rank_to_dict(ranks, names):
    ranks = np.abs(ranks)
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(np.array(ranks).reshape(14, 1)).ravel()
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks))

ranks = {}
features = ["x%s" % i for i in range(1, 15)]

ridge = Ridge(alpha=1.0)
ridge.fit(X, Y)
ranks["Ridge1"] = rank_to_dict(ridge.coef_, features)
ridge = Ridge(alpha=5.0)
ridge.fit(X, Y)
ranks["Ridge5"] = rank_to_dict(ridge.coef_, features)
ridge = Ridge(alpha=15)
ridge.fit(X, Y)
ranks["Ridge15"] = rank_to_dict(ridge.coef_, features)

rfg = RandomForestRegressor(max_depth=2, min_samples_leaf=1, min_impurity_decrease=0, ccp_alpha=0)
rfg.fit(X, Y)
ranks["RFR2"] = rank_to_dict(rfg.feature_importances_, features)

rfg = RandomForestRegressor(max_depth=4, min_samples_leaf=1, min_impurity_decrease=0, ccp_alpha=0)
rfg.fit(X, Y)
ranks["RFR4"] = rank_to_dict(rfg.feature_importances_, features)

rfg = RandomForestRegressor(max_depth=10, min_samples_leaf=1, min_impurity_decrease=0, ccp_alpha=0)
rfg.fit(X, Y)
ranks["RFR10"] = rank_to_dict(rfg.feature_importances_, features)

rfg = RandomForestRegressor(max_depth=10, min_samples_leaf=5, min_impurity_decrease=0, ccp_alpha=0)
rfg.fit(X, Y)
ranks["RFR1050"] = rank_to_dict(rfg.feature_importances_, features)

rfg = RandomForestRegressor(max_depth=16, min_samples_leaf=1, min_impurity_decrease=0, ccp_alpha=0)
rfg.fit(X, Y)
ranks["RFR1052"] = rank_to_dict(rfg.feature_importances_, features)

selector = SelectKBest(f_regression)
selector.fit(X, Y)
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

print('VALUES')
for r in ranks.items():
    print(r)
print('MEAN')
print(mean)


for i, (model_name, features) in enumerate(ranks.items()):
    subplot = plt.subplot(3, 3, i+1)
    subplot.set_title(model_name)
    subplot.bar(list(features.keys()), list(features.values()))


plt.show()
