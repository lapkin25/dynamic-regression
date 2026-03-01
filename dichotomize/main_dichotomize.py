import csv
import numpy as np
import math
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

def norm_prob(p):
    return min(max(p, 1e-6), 1 - 1e-6)


class BinaryModel:
    def __init__(self, x1_th, x2_th, y_th):
        self.x1_th = x1_th
        self.x2_th = x2_th
        self.y_th = y_th
        self.x1 = None
        self.x2 = None
        self.y = None

    def fit(self, x, y):
        assert(x.shape[1] == 2)
        self.x1 = np.where(x[:, 0] >= self.x1_th, 1, 0)
        self.x2 = np.where(x[:, 1] >= self.x2_th, 1, 0)
        self.y = np.where(y[:] >= self.y_th, 1, 0)

    def MI_score(self):
        N = len(self.y)
        N1 = np.sum(self.y)
        N0 = N - N1
        if N1 == 0 or N0 == 0:
            return 0.0

        P11_overall = norm_prob(np.sum((self.x1 == 1) & (self.x2 == 1)) / N)
        P10_overall = norm_prob(np.sum((self.x1 == 1) & (self.x2 == 0)) / N)
        P01_overall = norm_prob(np.sum((self.x1 == 0) & (self.x2 == 1)) / N)
        P00_overall = norm_prob(np.sum((self.x1 == 0) & (self.x2 == 0)) / N)

        P11_cond_1 = norm_prob(np.sum((self.x1 == 1) & (self.x2 == 1) & (self.y == 1)) / N1)
        P11_cond_0 = norm_prob(np.sum((self.x1 == 1) & (self.x2 == 1) & (self.y == 0)) / N0)
        P10_cond_1 = norm_prob(np.sum((self.x1 == 1) & (self.x2 == 0) & (self.y == 1)) / N1)
        P10_cond_0 = norm_prob(np.sum((self.x1 == 1) & (self.x2 == 0) & (self.y == 0)) / N0)
        P01_cond_1 = norm_prob(np.sum((self.x1 == 0) & (self.x2 == 1) & (self.y == 1)) / N1)
        P01_cond_0 = norm_prob(np.sum((self.x1 == 0) & (self.x2 == 1) & (self.y == 0)) / N0)
        P00_cond_1 = norm_prob(np.sum((self.x1 == 0) & (self.x2 == 0) & (self.y == 1)) / N1)
        P00_cond_0 = norm_prob(np.sum((self.x1 == 0) & (self.x2 == 0) & (self.y == 0)) / N0)

        H_Y = - (N1 / N) * math.log(N1 / N) - (N0 / N) * math.log(N0 / N)
        H_X = - P11_overall * math.log(P11_overall) - P10_overall * math.log(P10_overall) \
              - P01_overall * math.log(P01_overall) - P00_overall * math.log(P00_overall)

        H_X_cond_1 = - P11_cond_1 * math.log(P11_cond_1) - P10_cond_1 * math.log(P10_cond_1) \
                     - P01_cond_1 * math.log(P01_cond_1) - P00_cond_1 * math.log(P00_cond_1)
        H_X_cond_0 = - P11_cond_0 * math.log(P11_cond_0) - P10_cond_0 * math.log(P10_cond_0) \
                     - P01_cond_0 * math.log(P01_cond_0) - P00_cond_0 * math.log(P00_cond_0)

        H_X_cond_Y = (N1 / N) * H_X_cond_1 + (N0 / N) * H_X_cond_0

        mutual_information = H_X - H_X_cond_Y

        return mutual_information


class InterpretableModel:
    def __init__(self):
        self.bin_model = None
        self.x1_th = None
        self.x2_th = None
        self.y_th = None

    def fit(self, x, y):
        assert(x.shape[1] == 2)
        x1 = x[:, 0]
        x2 = x[:, 1]
        max_score = None
        best_y_th = None
        best_x1_th = None
        best_x2_th = None
        for y_th in np.linspace(np.min(y), np.max(y), 30):
            for x1_th in np.linspace(np.min(x1), np.max(x1), 30):
                for x2_th in np.linspace(np.min(x2), np.max(x2), 30):
                    bin_model = BinaryModel(x1_th, x2_th, y_th)
                    bin_model.fit(x, y)
                    score = bin_model.MI_score()
                    if max_score is None or score > max_score:
                        max_score = score
                        best_y_th, best_x1_th, best_x2_th = y_th, x1_th, x2_th
        self.x1_th = best_x1_th
        self.x2_th = best_x2_th
        self.y_th = best_y_th
        self.bin_model = BinaryModel(self.x1_th, self.x2_th, self.y_th)
        self.bin_model.fit(x, y)

    def MI_score(self):
        return self.bin_model.MI_score()


def plot_model(x1, x2, y, x1_th, x2_th, y_th):
#    plt.plot(x1[x2 >= x2_th], y[x2 >= x2_th], 'ro', alpha=0.5)
#    plt.plot(x1[x2 < x2_th], y[x2 < x2_th], 'bo', alpha=0.5)
    plt.plot(x1[(x2 >= x2_th) & (x1 >= x1_th)], y[(x2 >= x2_th) & (x1 >= x1_th)], 'ro', alpha=0.9)
    plt.plot(x1[(x2 >= x2_th) & (x1 < x1_th)], y[(x2 >= x2_th) & (x1 < x1_th)], 'ro', alpha=0.3)
    plt.plot(x1[(x2 < x2_th) & (x1 >= x1_th)], y[(x2 < x2_th) & (x1 >= x1_th)], 'bo', alpha=0.9)
    plt.plot(x1[(x2 < x2_th) & (x1 < x1_th)], y[(x2 < x2_th) & (x1 < x1_th)], 'bo', alpha=0.3)
    plt.xlabel("x(t)")
    plt.ylabel("x(t+1)")
    ax = plt.gca()
    ax.axline((np.min(x1), y_th), (np.max(x1), y_th))
    ax.axline((x1_th, np.min(y)), (x1_th, np.max(y)))
    plt.show()


print("Чтение данных из файла...", end='')
data, names, years = read_data()
print(" Прочитано")

ind1 = 0

print("Прогнозируемый ряд:", names[ind1])

# восстанавливаем зависимость x(t + 1) от x(t) и y(t)

x1 = data[:-1, ind1]  # ряд x(t)
y = data[1:, ind1]  # ряд x(t + 1)

for ind2 in range(6, 24):
#    if names[ind2] != "МаксШирота_янв":
#        continue

    print("Вспомогательный ряд:", names[ind2])
    x2 = data[:-1, ind2]  # ряд y(t)

    X = np.column_stack((x1, x2))
    model = InterpretableModel()
    model.fit(X, y)
    print("MI_score =", model.MI_score())
    print("   порог x(t) =", model.x1_th)
    print("   порог y(t) =", model.x2_th)
    print("   порог x(t+1) =", model.y_th)

#    plot_model(x1, x2, y, model.x1_th, model.x2_th, model.y_th)