import csv
import numpy as np
import math
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression


def read_data():
    """
    years - номера годов
    names - названия рядов с данными
    data - матрица, где строки - это годы, столбцы - это ряды
    """
    with open('winter.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        fields = next(reader)
        years = np.array(list(map(int, fields[1:])))
        data = None
        names = []
        for row in reader:
            names.append(row[0])
            data_row = np.array(list(map(float, row[1:])))
            if data is None:
                data = data_row
            else:
                data = np.vstack((data, data_row))
    return data.T, names, years


def membership(k, x, a):
    """
    Возвращает вектор мер принадлежности точки x классу с номером k (0, 1, ...)
    a - упорядоченный список центроидов классов
    """
    m = len(a)  # количество диапазонов
    if k == 0:
        if x <= a[0]:
            u_val = 1
        elif x > a[0] and x <= a[1]:
            u_val = (a[1] - x) / (a[1] - a[0])
        else:
            u_val = 0
    elif k == m - 1:
        if x >= a[m - 1]:
            u_val = 1
        elif x >= a[m - 2] and x < a[m - 1]:
            u_val = (x - a[m - 2]) / (a[m - 1] - a[m - 2])
        else:
            u_val = 0
    else:  # 0 < k < m - 1
        if x >= a[k - 1] and x <= a[k]:
            u_val = (x - a[k - 1]) / (a[k] - a[k - 1])
        elif x >= a[k] and x <= a[k + 1]:
            u_val = (a[k + 1] - x) / (a[k + 1] - a[k])
        else:
            u_val = 0
    return u_val


def obj_func(x1, x2, y, alpha, a, b, d):
    """
    :param x1: первый ряд-предиктор
    :param x2: второй ряд-предиктор
    :param y: выходной признак
    :param alpha: вертикальный центроид
    :param a: коэффициент при x1
    :param b: коэффициент при x2
    :param d: горизонтальные центроиды
    :return: значение целевой функции (энтропия) + матрица соответствия
    """

    # число наблюдений
    n = len(x1)
    # число диапазонов
    m = 3

    # интегральный показатель
    z = a * x1 + b * x2

    # центроиды диапазонов
    horiz_centroids = d
    vert_centroids = np.array([-alpha, 0, alpha])

    # расчет мер принадлежности
    u = np.zeros((n, m))  # меры принадлежности горизонтальным диапазонам
    v = np.zeros((n, m))  # меры принадлежности вертикальным диапазонам
    for i in range(n):
        for k in range(m):
            u[i, k] = membership(k, z[i], horiz_centroids)
            v[i, k] = membership(k, y[i], vert_centroids)

    # расчет матрицы соответствия
    mat = np.zeros((m, m))
    for s in range(m):
        for p in range(m):
            mat[s, p] = np.sum(u[:, s] * v[:, p]) / np.sum(u[:, s])

    # расчет целевой функции
    J = 0.0
    for i in range(n):
        for s in range(m):
            for p in range(m):
                J += u[i, s] * mat[s, p] * math.log(mat[s, p])
    J *= -(1 / n)

    return J, mat


def optimize_obj_func(x1, x2, y, alpha, a_init, b_init, d_init):
    """
    Оптимизация целевой функции
    :param x1: первый ряд-предиктор
    :param x2: второй ряд-предиктор
    :param y: выходной признак
    :param alpha: вертикальный центроид
    :return: a, b, d
    a - весовой коэффициент при x1
    b - весовой коэффициент при x2
    d - вектор горизонтальных центроидов
    """

    def f(params):
        a = params[0]
        b = params[1]
        d = params[2:]
        J, _ = obj_func(x1, x2, y, alpha, a, b, d)
        return J

    params_init = np.hstack(([a_init, b_init], d_init))
    res = minimize(f, params_init, method='Nelder-Mead')
    params = res.x
    a = params[0]
    b = params[1]
    d = params[2:]
    return a, b, d



print("Чтение данных из файла...", end='')
data, names, years = read_data()
print(" Прочитано")

x1 = data[:-1, 0]
x2 = data[:-1, 21]
y = data[1:, 0] - data[:-1, 0]


regr = LinearRegression()
regr.fit(np.vstack((x1, x2)).T, y)
a, b = regr.coef_
z = a * x1 + b * x2
d = np.array([np.min(z), (np.min(z) + np.max(z)) / 2, np.max(z)])


#a = 0
#b = 0
#d = [0, 0.1, 0.2]
alpha = 0.3

J, mat = obj_func(x1, x2, y, alpha, a, b, d)
print(J)
print(mat)

print("Оптимизация...", end='')
a, b, d = optimize_obj_func(x1, x2, y, alpha, a, b, d)
print(" Готово!")
print("a =", a)
print("b =", b)
print("d =", d)

J, mat = obj_func(x1, x2, y, alpha, a, b, d)
print(J)
print(mat)


#print(years)
#print(names)
#print(data)

