import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# Генерируем данные
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=777)
rand = np.random
rand.seed(777)
X += 1.2 * rand.uniform(size=X.shape)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# Функции
def generate_subplot(index):
    new_subplot = plt.subplot(1, 3, index)
    new_subplot.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    new_subplot.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.35)
    return new_subplot


def score_by_accuracy(model):
    y_predicted = model.predict(X_test)
    y_predicted_class = np.zeros(len(y_predicted))
    for i, val in enumerate(y_predicted):
        if val >= 0.5:
            y_predicted_class[i] = 1
        else:
            y_predicted_class[i] = 0
    print(accuracy_score(y_test, y_predicted_class))



# Генерируем сетку для рисования градиента
h = 0.02
x0_min, x0_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x1_min, x1_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, h), np.arange(x1_min, x1_max, h), sparse=False)

cm_bright = ListedColormap(['#FF0000', '#0000FF'])

# Линейная регрессия
current_subplot = generate_subplot(1)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

Z = lin_reg.predict(np.c_[xx0.ravel(), xx1.ravel()])
Z = Z.reshape(xx0.shape)
current_subplot.contourf(xx0, xx1, Z, cmap=cm.cool, alpha=.4)

# print(f'Coefficient of determination: {round(lin_reg.score(X_test, y_test), 4)}', f'Расчетные коэффициенты: {lin_reg.coef_}')
score_by_accuracy(lin_reg)


# Полиномиальная регрессия степени 4
current_subplot = generate_subplot(2)

polynomial_features = PolynomialFeatures(degree=4)
pipline = Pipeline([("Linear", polynomial_features), ("linear_regression", lin_reg)])
pipline.fit(X_train, y_train)

Z = pipline.predict(np.c_[xx0.ravel(), xx1.ravel()])
Z = Z.reshape(xx0.shape)
current_subplot.contourf(xx0, xx1, Z, cmap=cm.cool, alpha=.45)

# print(f'Polynomial: {round(pipline.score(X_test, y_test), 4)}')
score_by_accuracy(pipline)


# Гребневая полиносмиальная регрессия
current_subplot = generate_subplot(3)

ridge = Ridge(alpha=7.0)
polynomial_features = PolynomialFeatures(degree=4)
pipline = Pipeline([("polynomial_features", polynomial_features), ("ridge_regression", ridge)])
pipline.fit(X_train, y_train)

Z = pipline.predict(np.c_[xx0.ravel(), xx1.ravel()])
Z = Z.reshape(xx0.shape)
current_subplot.contourf(xx0, xx1, Z, cmap=cm.cool, alpha=.45)
# print(f'Ridge: {round(pipline.score(X_test, y_test), 4)}')
score_by_accuracy(pipline)


plt.show()
