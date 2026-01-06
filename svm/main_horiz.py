import csv
import numpy as np
import math
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay


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
#data = data[-20:, :]
#years = years[-20:]

ind1 = 28
#ind2 = 6

print("Прогнозируемый ряд:", names[ind1])
#print("Вспомогательный ряд:", names[ind2])

x1 = data[:-1, ind1]
y = data[1:, ind1]
#x2 = data[:-1, ind2]

#x2_threshold = np.mean(x2)

# TODO: собрать точки (x1, y); предсказать метки класса через условие x2 > x2_threshold (SVC)
# https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html

random_state = 42
C1 = 1
best_ind2 = None
global_mutual_information = -1.0
global_x2_threshold = None
for ind2 in range(6, 24):
    print("Вспомогательный ряд:", names[ind2])
    x2 = data[:-1, ind2]

    max_mutual_information = -1.0
    for x2_th in np.linspace(np.min(x2), np.max(x2)):
        X = np.column_stack((x1, y))
        Y = np.where(x2 > x2_th, 1, 0)

        N = len(Y)
        N1 = np.sum(Y)
        N0 = N - N1

        if N1 == 0 or N0 == 0:
            continue

        clf = svm.SVC(kernel="linear", probability=True, C=C1, random_state=random_state)
        clf.fit(X, Y)

        #print(X)
        #print(Y)
        #print(clf.decision_function(X))
        #print(clf.coef_)
        #print(clf.intercept_)
        #print(clf.predict_proba(X))

        p = clf.predict_proba(X)

        """
        # проба
        Z = np.where(y > np.mean(y), 1, 0)
        logist = LogisticRegression(max_iter=10000)
        logist.fit(y.reshape(-1, 1), Z)
        p = logist.predict_proba(y.reshape(-1, 1))
        """

        P1 = np.mean(p[Y == 1, 1])  # вероятность правильного предсказания класса "1"
        P0 = np.mean(p[Y == 0, 0])  # вероятность правильного предсказания класса "0"
        #print(f"P1 = {P1}, P0 = {P0}")

        P1_overall = np.mean(p[:, 1])  # P(X = 1)
        P0_overall = np.mean(p[:, 0])  # P(X = 0)

        mutual_information = -1.0
        best_y_th = None
        for y_th in np.linspace(np.min(y), np.max(y)):
            # проба
            M1 = np.sum(np.where(y > y_th, 1, 0))
            M0 = N - M1
            if M1 == 0 or M0 == 0:
                continue
            P1_overall = M1 / N
            P0_overall = M0 / N
            P1 = np.sum(np.where((y > y_th) & (Y == 1), 1, 0)) / N1
            P0 = np.sum(np.where((y <= y_th) & (Y == 0), 1, 0)) / N0
            P0 = min(max(P0, 1e-6), 1 - 1e-6)
            P1 = min(max(P1, 1e-6), 1 - 1e-6)


            H_Y = - (N1 / N) * math.log(N1 / N) - (N0 / N) * math.log(N0 / N)
            H_X = - P1_overall * math.log(P1_overall) - P0_overall * math.log(P0_overall)

            H_X_cond_1 = - P1 * math.log(P1) - (1 - P1) * math.log(1 - P1)
            H_X_cond_0 = - P0 * math.log(P0) - (1 - P0) * math.log(1 - P0)

            H_X_cond_Y = (N1 / N) * H_X_cond_1 + (N0 / N) * H_X_cond_0

            mutual_information0 = H_X - H_X_cond_Y
            if mutual_information0 > mutual_information:
                mutual_information = mutual_information0
                best_y_th = y_th



        def bin_entropy(p):
            return - p * math.log(p) - (1 - p) * math.log(1 - p)

        """
        # проверка
        H_Y_cond_X = P0_overall * bin_entropy((P0 * N0) / (P0 * N0 + (1 - P1) * N1)) + P1_overall * bin_entropy((P1 * N1) / (P1 * N1 + (1 - P0) * N0))
        mutual_information1 = H_Y - H_Y_cond_X
        print(mutual_information, mutual_information1)
        """

        # проверка с расстоянием К-Л
        #mutual_information = - (H_Y + (N1 / N) * math.log(P1) + (N0 / N) * math.log(P0))


        #print("x2_th =", x2_th, "MI =", mutual_information)
        if mutual_information > max_mutual_information:
            max_mutual_information = mutual_information
            x2_threshold = x2_th

    print("MI =", max_mutual_information)

    if max_mutual_information > global_mutual_information:
        global_mutual_information = max_mutual_information
        best_ind2 = ind2
        global_x2_threshold = x2_threshold


#print("x2_threshold =", x2_threshold)
print("Наилучший вспомогательный ряд:", names[best_ind2], "; порог =", global_x2_threshold)
x2 = data[:-1, best_ind2]
X = np.column_stack((x1, y))
Y = np.where(x2 > global_x2_threshold, 1, 0)
clf = svm.SVC(kernel="linear", probability=True, C=C1, random_state=random_state)
clf.fit(X, Y)
plt.scatter(X[:, 0], X[:, 1], c=Y) #, s=30, cmap=plt.cm.Paired)
ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    plot_method="contour",
    colors="k",
    levels=[-1, 0, 1],
    alpha=0.5,
    linestyles=["--", "-", "--"],
    ax=ax,
)
plt.show()

