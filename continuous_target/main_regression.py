import csv
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


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


print("Чтение данных из файла...", end='')
data, names, years = read_data()
print(" Прочитано")
data = data[-16:, :]
years = years[-16:]

ind1 = 4
#ind2 = 7

print("Прогнозируемый ряд:", names[ind1])
#print("Вспомогательный ряд:", names[ind2])

x1 = data[:-1, ind1]
y = data[1:, ind1]
#x2 = data[:-1, ind2]

"""
print("Baseline")
regr = LinearRegression()
regr.fit(x1.reshape(-1, 1), y)
w = regr.coef_[0]
w0 = regr.intercept_
print("w =", w, "w0 =", w0)
y_pred = w * x1 + w0
plt.plot(years, data[:, ind1])
plt.plot(years[1:], y_pred)
plt.show()
plt.plot(x1[x2 > x2_threshold], y[x2 > x2_threshold], 'bo', alpha=0.5)
plt.plot(x1[x2 <= x2_threshold], y[x2 <= x2_threshold], 'ro', alpha=0.5)
plt.xlabel("x(t)")
plt.ylabel("x(t+1)")
plt.show()
# TODO: изобразить синюю и красную линии регрессии
"""

global_err = None
global_x2_threshold = None
global_w_A = None
global_w0_A = None
global_w_B = None
global_w0_B = None
best_ind2 = None
for ind2 in range(6, 24):
    #if "Интен" in names[ind2] or "Долг" in names[ind2]:
    #    continue
    #if names[ind2] != "МаксИнтенс_фев":
    #    continue

    print("Вспомогательный ряд:", names[ind2])
    x2 = data[:-1, ind2]

    min_err = None
    x2_threshold = None
    best_w_A = None
    best_w0_A = None
    best_w_B = None
    best_w0_B = None
    for x2_th in np.linspace(np.min(x2), np.max(x2)):
        x1_A = x1[x2 > x2_th]
        y_A = y[x2 > x2_th]
        x1_B = x1[x2 <= x2_th]
        y_B = y[x2 <= x2_th]
        if len(x1_A) == 0 or len(x1_B) == 0:
            continue
        regr_A = LinearRegression()
        regr_A.fit(x1_A.reshape(-1, 1), y_A)
        ypred_A = regr_A.predict(x1_A.reshape(-1, 1))
        w_A = regr_A.coef_[0]
        w0_A = regr_A.intercept_
        regr_B = LinearRegression()
        regr_B.fit(x1_B.reshape(-1, 1), y_B)
        ypred_B = regr_B.predict(x1_B.reshape(-1, 1))
        w_B = regr_B.coef_[0]
        w0_B = regr_B.intercept_
        err = (mean_absolute_error(y_A, ypred_A) * len(y_A) + mean_absolute_error(y_B, ypred_B) * len(y_B)) / len(y)
        if min_err is None or err < min_err:
            min_err = err
            x2_threshold = x2_th
            best_w_A, best_w0_A, best_w_B, best_w0_B = w_A, w0_A, w_B, w0_B
    print("MSE =", min_err)
    if global_err is None or min_err < global_err:
        global_err = min_err
        best_ind2 = ind2
        global_x2_threshold = x2_threshold
        global_w_A, global_w0_A, global_w_B, global_w0_B = best_w_A, best_w0_A, best_w_B, best_w0_B
print("Наилучший вспомогательный ряд:", names[best_ind2], "; порог =", global_x2_threshold)
x2 = data[:-1, best_ind2]
plt.axline(xy1=(0, global_w0_A), slope=global_w_A, color='b')
plt.axline(xy1=(0, global_w0_B), slope=global_w_B, color='r')
plt.plot(x1[x2 > global_x2_threshold], y[x2 > global_x2_threshold], 'bo', alpha=0.5)
plt.plot(x1[x2 <= global_x2_threshold], y[x2 <= global_x2_threshold], 'ro', alpha=0.5)
plt.xlabel("x(t)")
plt.ylabel("x(t+1)")
plt.show()
y_pred = np.zeros_like(y)
for i in range(len(y)):
    if x2[i] > global_x2_threshold:
        y_pred[i] = global_w_A * x1[i] + global_w0_A
    else:
        y_pred[i] = global_w_B * x1[i] + global_w0_B
plt.plot(years, data[:, ind1])
plt.plot(years[1:], y_pred)
plt.show()