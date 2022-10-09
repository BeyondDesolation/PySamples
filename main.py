import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=0, n_clusters_per_class=1)
# rng = np.random.RandomState(2)
# X += 2 * rng.uniform(size=X.shape)
# linearly_dataset = (X, y)
#
# moon_dataset = make_moons(noise=0.3, random_state=0)
# circles_dataset = make_circles(noise=0.2, factor=0.5, random_state=1)
# datasets = [moon_dataset, circles_dataset, linearly_dataset]
#
# cm_bright = ListedColormap(['#FF0000', '#0000FF'])
#
# i = 0
# for ds_cnt, ds in enumerate(datasets):
#     i += 1
#     X, y = ds
#     X = StandardScaler().fit_transform(X)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
#     current_subplot = plt.subplot(1, 3, i)
#     current_subplot.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
#     current_subplot.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
#
# plt.show()

# X, y = make_classification(n_samples=50, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1)
# rng = np.random.RandomState(2)
# # X += 2 * rng.uniform(size=X.shape)
# # linearly_dataset = (X, y)
#
# X = StandardScaler().fit_transform(X)
# lin_reg = LinearRegression().fit(X, y)
# lin_predict = lin_reg.predict(X)
# # print(lin_predict)
# print(lin_reg.score(X, y), lin_reg.coef_, lin_reg.intercept_)
#
# h = .02
# x0_min, x0_max = X[:, 0].min() - .5, X[:, 0].max() + .5
# x1_min, x1_max = X[:, 1].min() - .5, X[:, 1].max() + .5
# xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, h), np.arange(x1_min, x1_max, h))
# Z = lin_reg.predict(np.c_[xx0.ravel(), xx1.ravel()])
# Z = Z.reshape(xx0.shape)
# print(Z)
# cm_bright = ListedColormap(['#FF0000', '#0000FF'])
#
# plt.contourf(xx0, xx1, Z, cmap=cm_bright, alpha=.5)
#
# # plt.set_xlim(xx0.min(), xx0.max())
# # plt.set_ylim(xx0.min(), xx1.max())
# # plt.set_xticks(())
# # plt.set_yticks(())
#
# # subplot = plt.subplot(1, 1, 1)
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
# plt.show()

# Генерируем данные
X, y = make_classification(n_samples=50, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

# Генерируем сетку для рисования градиента
h = .02
x0_min, x0_max = X[:, 0].min() - .5, X[:, 0].max() + .5
x1_min, x1_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, h), np.arange(x1_min, x1_max, h))

cm_bright = ListedColormap(['#FF0000', '#0000FF'])

# Линейная регрессия
current_subplot = plt.subplot(1, 3, 1)
current_subplot.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
current_subplot.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.35)

lin_reg = LinearRegression().fit(X_train, y_train)

Z = lin_reg.predict(np.c_[xx0.ravel(), xx1.ravel()])
Z = Z.reshape(xx0.shape)
current_subplot.contourf(xx0, xx1, Z, cmap=cm_bright, alpha=.15)
print(lin_reg.score(X, y), lin_reg.coef_, lin_reg.intercept_)

# Полиномиальная регрессия степени 4
current_subplot = plt.subplot(1, 3, 2)
current_subplot.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
current_subplot.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.35)

# pypline

print(X_train)
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X_train)
print(X_poly)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y_train)

# Z = pol_reg.predict(np.c_[xx0.ravel(), xx1.ravel()])
# Z = Z.reshape(xx0.shape)
# current_subplot.contourf(xx0, xx1, Z, cmap=cm_bright, alpha=.15)

# Z = pol_reg.predict(X_train)
# print(Z)
plt.show()
