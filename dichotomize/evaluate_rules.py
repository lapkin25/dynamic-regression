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
    with open('../cyclones_fareast.csv', newline='') as csvfile:
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


def eval_rules(rules):
    N = len(rules[0][0])
    total_covered = np.zeros(N)
    for i, (rule_left, rule_right) in enumerate(rules):
        coverage = np.sum(rule_left) / N
        precision = np.sum(np.logical_and(rule_left, rule_right)) / np.sum(rule_left)
        print("Правило", i + 1)
        print("Покрытие %.2f%%, Точность %.2f%%" % (coverage * 100, precision * 100))
        total_covered = np.logical_or(total_covered, rule_left)
    total_coverage = np.sum(total_covered) / N
    print("Общее покрытие %.2f%%" % (total_coverage * 100))



print("Чтение данных из файла...", end='')
data, names, years = read_data()
print(" Прочитано")

ind1 = 0
ind2 = 13
print("Прогнозируемый ряд:", names[ind1])
print("Вспомогательный ряд:", names[ind2])
x = data[:-1, ind1]  # ряд x(t)
y = data[:-1, ind2]  # ряд y(t)
z = data[1:, ind1]  # ряд x(t + 1)
rule_left_1 = (x < 19.5) & (y > 41.2)
rule_right_1 = (z > 17.8)
rule_left_2 = (x < 25.5) & (y < 41.2)
rule_right_2 = (z < 17.8)
eval_rules([(rule_left_1, rule_right_1), (rule_left_2, rule_right_2)])

ind1 = 1
ind2 = 12
print("Прогнозируемый ряд:", names[ind1])
print("Вспомогательный ряд:", names[ind2])
x = data[:-1, ind1]  # ряд x(t)
y = data[:-1, ind2]  # ряд y(t)
z = data[1:, ind1]  # ряд x(t + 1)
rule_left_1 = (x < 32) & (y > 107)
rule_right_1 = (z < 16.7)
rule_left_2 = (x < 14.6) & (y < 107)
rule_right_2 = (z > 16.7)
eval_rules([(rule_left_1, rule_right_1), (rule_left_2, rule_right_2)])

ind1 = 2
ind2 = 14
print("Прогнозируемый ряд:", names[ind1])
print("Вспомогательный ряд:", names[ind2])
x = data[:-1, ind1]  # ряд x(t)
y = data[:-1, ind2]  # ряд y(t)
z = data[1:, ind1]  # ряд x(t + 1)
rule_left_1 = (x > 16.1) & (y > 1003.7)
rule_right_1 = (z < 19.1)
rule_left_2 = (x > 18) & (y < 1003.7)
rule_right_2 = (z > 19.1)
eval_rules([(rule_left_1, rule_right_1), (rule_left_2, rule_right_2)])

ind1 = 3
ind2 = 13
print("Прогнозируемый ряд:", names[ind1])
print("Вспомогательный ряд:", names[ind2])
x = data[:-1, ind1]  # ряд x(t)
y = data[:-1, ind2]  # ряд y(t)
z = data[1:, ind1]  # ряд x(t + 1)
rule_left_1 = (x < 208) & (y > 38)
rule_right_1 = (z > 106)
rule_left_2 = (x < 208) & (y < 38)
rule_right_2 = (z < 106)
eval_rules([(rule_left_1, rule_right_1), (rule_left_2, rule_right_2)])

ind1 = 4
ind2 = 14
print("Прогнозируемый ряд:", names[ind1])
print("Вспомогательный ряд:", names[ind2])
x = data[:-1, ind1]  # ряд x(t)
y = data[:-1, ind2]  # ряд y(t)
z = data[1:, ind1]  # ряд x(t + 1)
rule_left_1 = (x < 156) & (y > 1003)
rule_right_1 = (z > 151.7)
rule_left_2 = (x < 166) & (y < 1003)
rule_right_2 = (z < 151.7)
eval_rules([(rule_left_1, rule_right_1), (rule_left_2, rule_right_2)])

ind1 = 5
ind2 = 12
print("Прогнозируемый ряд:", names[ind1])
print("Вспомогательный ряд:", names[ind2])
x = data[:-1, ind1]  # ряд x(t)
y = data[:-1, ind2]  # ряд y(t)
z = data[1:, ind1]  # ряд x(t + 1)
rule_left_1 = (x < 547) & (y > 109)
rule_right_1 = (z < 349.2)
rule_left_2 = (x > 134) & (y < 109)
rule_right_2 = (z > 349.2)
eval_rules([(rule_left_1, rule_right_1), (rule_left_2, rule_right_2)])
