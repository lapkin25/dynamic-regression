import csv
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def read_data():
    """
    years - номера годов
    names - названия рядов с данными
    data - матрица, где строки - это годы, столбцы - это ряды
    """
    with open('cyclones_fareast.csv', newline='') as csvfile:
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

def norm_prob(p):
    return min(max(p, 1e-6), 1 - 1e-6)


class PiecewiseModel:
    def __init__(self):
        self.linreg1 = None
        self.linreg2 = None
        self.c = None
        self.a1 = None
        self.a2 = None
        self.rmse = None

    def fit(self, x_, y):
        min_rmse = None
        best_c = None
        best_a1 = None
        best_a2 = None
        x = x_[:, 0].T
        for c in np.linspace(np.min(x), np.max(x), 100)[1:-1]:
            linreg1 = LinearRegression(fit_intercept=False)
            linreg2 = LinearRegression(fit_intercept=False)
            x1 = x[x < c] - c
            y1 = y[x < c]
            x2 = x[x >= c] - c
            y2 = y[x >= c]
            linreg1.fit(x1.reshape(-1, 1), y1)
            linreg2.fit(x2.reshape(-1, 1), y2)
            mse = (np.sum((linreg1.predict(x1.reshape(-1, 1)) - y1) ** 2) +
                   np.sum((linreg2.predict(x2.reshape(-1, 1)) - y2) ** 2)) / len(x)
            rmse = math.sqrt(mse)

            if min_rmse is None or rmse < min_rmse:
                min_rmse = rmse
                best_c = c
                best_a1 = linreg1.coef_[0]
                best_a2 = linreg2.coef_[0]

        self.c = best_c
        self.rmse = min_rmse
        self.a1 = best_a1
        self.a2 = best_a2



def plot_model(x, y, c, a1, a2):
    plt.plot(x, y, 'o')
    plt.xlabel("y(t)")
    plt.ylabel("x(t+1) - x(t)")
    plt.plot([np.min(x), c], [a1 * (np.min(x) - c), 0], 'r')
    plt.plot([c, np.max(x)], [0, a2 * (np.max(x) - c)], 'r')
    plt.show()


print("Чтение данных из файла...", end='')
data, names, years = read_data()
print(" Прочитано")

ind1 = 5

print("Прогнозируемый ряд:", names[ind1])

# восстанавливаем зависимость x(t + 1) от x(t) и y(t)

x1 = data[:-1, ind1]  # ряд x(t)
y = data[1:, ind1]  # ряд x(t + 1)

for ind2 in range(12, 15):
    #if names[ind2] != "АлеутДолгота_фев":
    #    continue

    print("Вспомогательный ряд:", names[ind2])
    x2 = data[:-1, ind2]  # ряд y(t)

    #X = np.column_stack((x1, x2))
    model = PiecewiseModel()
    model.fit(x2.reshape(-1, 1), y - x1)
    print("   RMSE =", model.rmse)
    print("   c =", model.c)
    print("   a1 =", model.a1)
    print("   a2 =", model.a2)

    plot_model(x2, y - x1, model.c, model.a1, model.a2)
