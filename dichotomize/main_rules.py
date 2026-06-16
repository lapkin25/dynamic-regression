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


# покрывает ли условие правила данную точку
def rule_covers(rule, x_val):
    return (rule['dir'] == '>' and x_val >= rule['x_th']) or (rule['dir'] == '<' and x_val <= rule['x_th'])


class OneColorRulesModel:
    def __init__(self, p0):
        self.p0 = p0
        self.rules = []

    def extract_rules(self, y_val, x, y):
        # правило вида x >= x_th
        best_x_th = None
        for x_th in np.linspace(np.min(x), np.max(x), 50):
            if np.all(x < x_th):
                continue
            N1 = np.sum(y[x >= x_th])
            N0 = np.sum(1 - y[x >= x_th])
            if y_val == 1:
                precision = N1 / (N1 + N0)
            else:
                precision = N0 / (N1 + N0)
            if precision >= self.p0:
                best_x_th = x_th
                break
        if best_x_th is not None:
            rule = {'y_val': y_val, 'dir': '>', 'x_th': best_x_th}
            self.rules.append(rule)

        # правило вида x <= x_th
        best_x_th = None
        for x_th in np.linspace(np.min(x), np.max(x), 50)[::-1]:
            if np.all(x > x_th):
                continue
            N1 = np.sum(y[x <= x_th])
            N0 = np.sum(1 - y[x <= x_th])
            if y_val == 1:
                precision = N1 / (N1 + N0)
            else:
                precision = N0 / (N1 + N0)
            if precision >= self.p0:
                best_x_th = x_th
                break
        if best_x_th is not None:
            rule = {'y_val': y_val, 'dir': '<', 'x_th': best_x_th}
            self.rules.append(rule)

    # x - непрерывная переменная (одна координата)
    # y - бинарная переменная
    # извлекаем правило для прогноза y = 1 или y = 0 по значению x
    def fit(self, x, y):
        if len(x) == 0:
            return
        self.extract_rules(1, x, y)
        self.extract_rules(0, x, y)

    def print_rules(self):
        for rule in self.rules:
            print(f"x {rule['dir']} {rule['x_th']} => y = {rule['y_val']}")


class TwoColorRulesModel:
    def __init__(self, p0, x2_th, y_th):
        self.p0 = p0
        self.x2_th = x2_th
        self.y_th = y_th
        self.model_red = None
        self.model_blue = None

    def fit(self, x, y):
        assert(x.shape[1] == 2)
        x1 = x[:, 0]
        x2 = x[:, 1]
        colors = (x2 >= self.x2_th)
        # color = 1 => точка красная;  color = 0 => точка синяя
        targets = (y >= self.y_th)
        # target = 1 => точка вверху;  target = 0 => точка внизу

        # цель - извлечь из данных правила для прогнозирования таргета
        #   по отдельности для красного и синего цветов;
        #   применяется идея метода Anchors в XAI:
        #   максимизируем покрытие правилом при требуемом уровне точности

        model_red = OneColorRulesModel(self.p0)
        model_red.fit(x1[colors == 1], targets[colors == 1])
        model_blue = OneColorRulesModel(self.p0)
        model_blue.fit(x1[colors == 0], targets[colors == 0])

        self.model_red = model_red
        self.model_blue = model_blue

    # какая доля (от 0 до 1) всех точек покрыта правилами
    def score(self, x):
        points_covered = 0
        for i in range(len(x)):
            val = x[i, 0]
            color = (x[i, 1] >= self.x2_th)
            if color == 1 and any([rule_covers(rule, val) for rule in self.model_red.rules]) or \
                    color == 0 and any([rule_covers(rule, val) for rule in self.model_blue.rules]):
                # точка покрыта правилами
                points_covered += 1
        return points_covered / len(x)


class RulesModel:
    def __init__(self, p0):
        self.y0 = None
        self.model = None
        self.p0 = p0

    def plot(self, x, y):
        x2_th = self.model.x2_th
        y_th = self.model.y_th
        plt.plot(x1[x[:, 1] >= x2_th], y[x[:, 1] >= x2_th], 'ro', alpha=0.9)
        plt.plot(x1[x[:, 1] < x2_th], y[x[:, 1] < x2_th], 'bo', alpha=0.9)

        plt.xlabel("x(t)")
        #plt.ylabel("x(t+1) - x(t)")
        plt.ylabel("x(t+1)")
        ax = plt.gca()
        ax.axline((np.min(x1), y_th), (np.max(x1), y_th), c='k')
        for rule in self.model.model_red.rules:
            x1_th = rule['x_th']
            ax.axline((x1_th, np.min(y)), (x1_th, np.max(y)), c='r')
        for rule in self.model.model_blue.rules:
            x1_th = rule['x_th']
            ax.axline((x1_th, np.min(y)), (x1_th, np.max(y)), c='b')
        plt.show()

    def fit(self, x, y):
        assert(x.shape[1] == 2)
        x1 = x[:, 0]
        x2 = x[:, 1]
        max_score = None
        best_y0 = None
        best_model = None
        for y0 in np.linspace(np.min(x2), np.max(x2), 50):
            model = TwoColorRulesModel(self.p0, y0, np.mean(y))  #0.0)
            model.fit(x, y)
            score = model.score(x)
            if max_score is None or score > max_score:
                max_score = score
                best_y0 = y0
                best_model = model
        self.y0 = best_y0
        self.model = best_model

        print(f"Покрытие правилами = {max_score * 100} %")
        print(f"y(t) > {self.y0}:")
        self.model.model_red.print_rules()
        print(f"y(t) < {self.y0}:")
        self.model.model_blue.print_rules()


print("Чтение данных из файла...", end='')
data, names, years = read_data()
print(" Прочитано")

p0 = 0.75
ind1 = 5

print("Прогнозируемый ряд:", names[ind1])

# восстанавливаем зависимость x(t + 1) от x(t) и y(t)

x1 = data[:-1, ind1]  # ряд x(t)
y = data[1:, ind1]  # ряд x(t + 1)
#z = y - x1
z = y

for ind2 in range(12, 15):
    #if names[ind2] != "АлеутДолгота_фев":
    #    continue

    print("Вспомогательный ряд:", names[ind2])
    x2 = data[:-1, ind2]  # ряд y(t)

    X = np.column_stack((x1, x2))
    model = RulesModel(p0)
    model.fit(X, z)
    model.plot(X, z)
