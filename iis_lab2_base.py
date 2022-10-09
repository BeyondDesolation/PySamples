from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Генерируем исходные данные: 750 строк-наблюдений и 14 столбцов-признаков
np.random.seed(0)
size = 750
X = np.random.uniform(0, 1, (size, 14))

# Задаем функцию-выход: регрессионную проблему Фридмана
Y = (10 * np.sin(np.pi*X[:, 0]*X[:, 1]) + 20*(X[:, 2] - .5)**2 + 10 * X[:, 3] + 5*X[:, 4]**5 + np.random.normal(0, 1))

# Добавляем зависимость признаков
X[:, 10:] = X[:, :4] + np.random.normal(0, 0.025, (size, 4))

# Линейная модель
lr = LinearRegression()
lr.fit(X, Y)

# Гребневая модель
ridge = Ridge(alpha=7)
ridge.fit(X, Y)

# Лассо
lasso = Lasso(alpha=.05)
lasso.fit(X, Y)

names = ["x%s" % i for i in range(1, 15)]
ranks = {}

def rank_to_dict(ranks, names):
    ranks = np.abs(ranks)
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(np.array(ranks).reshape(14, 1)).ravel()
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks))


ranks["Linear reg"] = rank_to_dict(lr.coef_, names)
ranks["Ridge"] = rank_to_dict(ridge.coef_, names)
ranks["Lasso"] = rank_to_dict(lasso.coef_, names)

for r in ranks.items():
    print(r)

mean = {}

for item in ranks.values():
    for key, value in item.items():
        if key not in mean:
            mean[key] = 0
        mean[key] += value

for key, value in mean.items():
    res = value / len(ranks)
    mean[key] = round(res, 2)


sorted_keys = sorted(mean, key=mean.get, reverse=True)
sorted_dict = {}
for key in sorted_keys:
    sorted_dict[key] = mean[key]

print("MEAN")
print(mean)
