import numpy as np

# numpy
# Любой numpy array имеет свойства:
# shape - кортеж, описывающий размер по каждому измерению.
# dtype - объект, описывающий тип данных в массиве
# ndim - кол-во измерений tolist() - конвертирует в python list

# array vs list
# использует меньше памяти, больше функционал, требуют однородности данных.

# СОЗДАНИЕ
data = [1, 2, 3, 4, 5]
arr = np.array(data)
arr = np.array([1, 2, 3, 4, 5], dtype=np.float64)
arr = np.array([1.9, 2, 3, 4, 5])
arr = np.arange(10, 200, 20)
arr = np.linspace(10.0, 40.0, 15)
arr = np.random.random(5) * 5 + 10
# Изменение типа
# arr = arr.astype(dtype=np.int64)

# Преобразования
arr = np.sqrt(arr)
arr = np.sin(arr)
arr = np.exp(arr)
arr = np.log(arr)

arr2 = np.random.random(5)
arr3 = arr2 + arr

# print(arr3)
# print(arr3 > 0)

# Срезы
arr = np.arange(1, 10, 1)
print(arr[0])
print(arr[0:2]) # с 0 по 2 элементы
print(arr[::-1]) # реверс. От первого до последнего с шагом -1
arr[0:4:2] = 0
print(arr) 
print(arr[(arr < 6) & (arr > 0)])

arr[0:5] = np.arange(100, 110, 2)
print(arr)
print(arr, arr.shape, arr.dtype, arr.ndim, arr.size, len(arr))

print("asddddddddddddddddddddddddddddd")

# Матрицы
mat = np.array([(1, 2, 5), (4, 5, 6)])
mat = np.array([[1, 2, 5], [4, 5, 6]])
mat = mat.reshape(1, 6)
mat = np.random.random((3, 3))
arr = np.random.random(8)
print(arr)
mat = arr.reshape(2, 4)
mat = np.resize(mat, (2, 2))

mat = np.zeros((2, 3))
mat = np.ones((2, 3))
mat = np.eye(5)
mat = np.full((2, 3), 9)
print(mat, mat.shape, mat.dtype, mat.ndim, mat.size)
