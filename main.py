import csv
import numpy as np
import math
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
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
                if mat[s, p] > 0:
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
        d = np.sort(params[2:])
        J, _ = obj_func(x1, x2, y, alpha, a, b, d)
        return J

    params_init = np.hstack(([a_init, b_init], d_init))
    res = minimize(f, params_init, method='Nelder-Mead')
    params = res.x
    a = params[0]
    b = params[1]
    d = np.sort(params[2:])
    return a, b, d


def obj_func_cat(x1, x2, y, alpha, d, q):
    """
    :param x1: первый ряд-предиктор
    :param x2: второй ряд-предиктор
    :param y: выходной признак
    :param alpha: вертикальный центроид
    :param d: центроиды для x1
    :param q: центроиды для x2
    :return: значение целевой функции (энтропия) + матрица соответствия
    """
    # число наблюдений
    n = len(x1)
    # число диапазонов признака x1
    m1 = 3
    # число диапазонов признака x2
    m2 = 2
    # число диапазонов выходного признака
    m = 3

    # центроиды диапазонов
    horiz_centroids_1 = d
    horiz_centroids_2 = q
    vert_centroids = np.array([-alpha, 0, alpha])

    # расчет мер принадлежности
    u = np.zeros((n, m1))  # меры принадлежности диапазонам признака x1
    v = np.zeros((n, m2))  # меры принадлежности диапазонам признака x2
    w = np.zeros((n, m))  # меры принадлежности диапазонам выходного признака
    for i in range(n):
        for k in range(m1):
            u[i, k] = membership(k, x1[i], horiz_centroids_1)
    for i in range(n):
        for k in range(m2):
            v[i, k] = membership(k, x2[i], horiz_centroids_2)
    for i in range(n):
        for k in range(m):
            w[i, k] = membership(k, y[i], vert_centroids)

    # расчет матрицы соответствия
    mat = np.zeros((m1, m2, m))
    for s in range(m1):
        for p in range(m2):
            for r in range(m):
                if np.sum(u[:, s] * v[:, p]) == 0:
                    mat[s, p, r] = 1e10
                else:
                    mat[s, p, r] = np.sum(u[:, s] * v[:, p] * w[:, r]) / np.sum(u[:, s] * v[:, p])

    # расчет целевой функции
    J = 0.0
    for i in range(n):
        for s in range(m1):
            for p in range(m2):
                for r in range(m):
                    if mat[s, p, r] == 1e10:
                        J -= 1e10
                    if mat[s, p, r] > 0:
                        J += u[i, s] * v[i, p] * mat[s, p, r] * math.log(mat[s, p, r])
    J *= -(1 / n)

    return J, mat


def optimize_obj_func_cat(x1, x2, y, alpha, d_init, q_init):
    """
    Оптимизация целевой функции
    :param x1: первый ряд-предиктор
    :param x2: второй ряд-предиктор
    :param y: выходной признак
    :param alpha: вертикальный центроид
    :return: d, q
    d - вектор центроидов признака x1
    q - вектор центроидов признака x2
    """

    def f(params):
        d = np.sort(params[:m1])
        q = np.sort(params[m1:])
        J, _ = obj_func_cat(x1, x2, y, alpha, d, q)
        return J

    m1 = len(d_init)
    m2 = len(q_init)
    params_init = np.hstack((d_init, q_init))
    res = minimize(f, params_init, method='Nelder-Mead')
    params = res.x
    d = np.sort(params[:m1])
    q = np.sort(params[m1:])
    return d, q


print("Чтение данных из файла...", end='')
data, names, years = read_data()
print(" Прочитано")

ind1 = 0
ind2 = 12

print("Прогнозируемый ряд:", names[ind1])
print("Вспомогательный ряд:", names[ind2])

x1 = data[:-1, ind1]
x2 = data[:-1, ind2]
y = data[1:, ind1] - data[:-1, ind1]


print("Решающее дерево (глубина 2)...")
clf = DecisionTreeClassifier(random_state=0, max_depth=2)  #, criterion='entropy')
clf.fit(np.vstack((x1, x2)).T, np.where(y > 0, 1, 0))
print("accuracy =", clf.score(np.vstack((x1, x2)).T, np.where(y > 0, 1, 0)))
plot_tree(clf, proportion=True)
plt.show()



y_bin = np.where(y > 0, 1, 0)
grid1 = np.linspace(np.min(x1), np.max(x1), 50, endpoint=False)
grid2 = np.linspace(np.min(x2), np.max(x2), 50, endpoint=False)
max_auc = 0.0
best_A = None
best_B = None
print("Вычисляем оптимальные пороги...")
for A in grid1:
    for B in grid2:
        x1_bin = np.where(x1 <= A, 1, 0)
        x2_bin = np.where(x2 <= B, 1, 0)
        model = LogisticRegression(max_iter=10000)
        model.fit(np.vstack((x1_bin, x2_bin)).T, y_bin)
        p = model.predict_proba(np.vstack((x1_bin, x2_bin)).T)[:, 1]
        auc = roc_auc_score(y_bin, p)
        if auc > max_auc:
            max_auc = auc
            best_A = A
            best_B = B
        #print(A, B, auc)
print("max_auc =", max_auc)
x1_bin = np.where(x1 <= best_A, 1, 0)
x2_bin = np.where(x2 <= best_B, 1, 0)
model = LogisticRegression(max_iter=10000)
model.fit(np.vstack((x1_bin, x2_bin)).T, y_bin)
coef_A = model.coef_.ravel()[0]
coef_B = model.coef_.ravel()[1]
print(f"Модель: {coef_A} * [x <= {best_A}] + {coef_B} * [y <= {best_B}] + {model.intercept_}")
#y_bin_pred = np.zeros(len(y_bin))
#for i in range(len(y_bin)):
y_bin_pred = model.predict(np.vstack((x1_bin, x2_bin)).T)
#y_bin_pred = x1_bin  # baseline!
print("accuracy =", accuracy_score(y_bin, y_bin_pred))
cm = confusion_matrix(y_bin, y_bin_pred)
tp = cm[1, 1]
tn = cm[0, 0]
fp = cm[0, 1]
fn = cm[1, 0]
print("sensitivity =", tp / (tp + fn))
print("specificity =", tn / (tn + fp))



regr = LinearRegression()
regr.fit(np.vstack((x1, x2)).T, y)
a, b = regr.coef_
print("a =", a, "b =", b)
z = a * x1 + b * x2
d = np.array([np.min(z), (np.min(z) + np.max(z)) / 2, np.max(z)])


y_bin = np.where(y > 0, 1, 0)
log_reg = LogisticRegression()
log_reg.fit(np.vstack((x1, x2)).T, y_bin)
y_pred = log_reg.predict(np.vstack((x1, x2)).T)
acc = accuracy_score(y_bin, y_pred)
print("accuracy (log) =", acc)


# выбираем оптимальный порог для z - бинарная классификация
grid = np.linspace(np.min(z), np.max(z), 100, endpoint=False)
y_bin = np.where(y > 0, 1, 0)
max_acc = 0.0
best_cut = None
for cut in grid:
    y_pred = np.where(z <= cut, 0, 1)
    acc = accuracy_score(y_bin, y_pred)
    if acc > max_acc:
        max_acc = acc
        best_cut = cut
print("accuracy (lin) =", max_acc, "при пороге z =", best_cut)

plt.plot(z, y, 'bo', alpha=0.5)
plt.plot([np.min(z), np.max(z)], [0.0, 0.0], 'g', linestyle='dashed')
plt.plot([best_cut, best_cut], [np.min(y), np.max(y)], 'r', linestyle='dashed')
plt.xlabel("ax(t) + by(t)")
plt.ylabel("x(t+1) - x(t)")
plt.savefig("1.png", dpi=300)
plt.show()



#a = 0
#b = 0
#d = [0, 0.1, 0.2]
alpha = 0.25

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


z = a * x1 + b * x2

plt.plot(z, y, 'bo', alpha=0.5)
# TODO: найти ординаты как средние значения
#plt.plot(np.hstack(([np.min(z)], d, [np.max(z)])), [-alpha, -alpha, 0, alpha, alpha], 'k', linewidth=3)
plt.plot([np.min(z), np.max(z)], [0.0, 0.0], 'g', linestyle='dashed')
plt.plot([d[1], d[1]], [np.min(y), np.max(y)], 'r', linestyle='dashed')

plt.xlabel("z(t)")
plt.ylabel("x(t+1) - x(t)")
plt.show()


fig, ax = plt.subplots()
ax.plot(years, data[:, ind1])
cnt_g = 0
cnt_r = 0
for i, (xc, yc) in enumerate(zip(years[:-1], data[:-1, ind1])):
    if z[i] < d[0]:
        pred = '-'
    elif z[i] < d[1]:
        if mat[1, 0] == np.max(mat[1, :]):
            pred = '-'
        elif mat[1, 2] == np.max(mat[1, :]):
            pred = '+'
        else:
            pred = '0'
    else:
        pred = '+'
    if y[i] > 0 and pred == '-' or y[i] < 0 and pred == '+':
        color = 'r'
        cnt_r += 1
    else:
        color = 'g'
        cnt_g += 1
    if pred == '+':
        s = '+'
    elif pred == '-':
        s = '–'
    else:
        s = '0'
    if color == 'r':
        s += '!'
    ax.text(xc, yc, s, fontsize=12, color=color)
plt.savefig("2.png", dpi=300)
plt.show()

print("accuracy =", cnt_g / (cnt_g + cnt_r))
# TODO: какая будет точность, если найти a, b из линейной регрессии?

# TODO: рассчитать вероятности отнесения к классам


print("Второй нечеткий метод")

d = np.array([np.min(x1), (np.min(x1) + np.max(x1)) / 2, np.max(x1)])
q = np.array([(np.min(x2) + np.max(x2)) / 2, (np.min(x2) + np.max(x2)) / 2])

J, mat = obj_func_cat(x1, x2, y, alpha, d, q)
print(J)
print(mat)

print("Оптимизация...", end='')
d, q = optimize_obj_func_cat(x1, x2, y, alpha, d, q)
print(" Готово!")
print("d =", d)
print("q =", q)

J, mat = obj_func_cat(x1, x2, y, alpha, d, q)
print(J)
print(mat)

y_bin = np.where(y > 0, 1, 0)
y_bin_pred = np.zeros_like(x1, dtype=int)
for i in range(len(x1)):
    pred_classes = np.zeros(3)
    for s in range(3):
        u = membership(s, x1[i], d)
        for p in range(2):
            v = membership(p, x2[i], q)
            for r in range(3):
                pred_classes[r] += u * v * mat[s, p, r]
    #print(pred_classes)
    if pred_classes[0] > pred_classes[2]:
        y_bin_pred[i] = 0
    else:
        y_bin_pred[i] = 1
acc = accuracy_score(y_bin, y_bin_pred)
print("accuracy =", acc)


"""
print("Оптимизируем для разных пар рядов...")
J_mat = np.zeros((len(names), len(names)))
for ind1 in list(range(0, 6)) + list(range(24, 26)):
    for ind2 in range(6, len(names)):
        print(names[ind1], "через", names[ind2], end='')

        x1 = data[:-1, ind1]
        x2 = data[:-1, ind2]
        y = data[1:, ind1] - data[:-1, ind1]

        regr = LinearRegression()
        regr.fit(np.vstack((x1, x2)).T, y)
        a, b = regr.coef_
        z = a * x1 + b * x2
        d = np.array([np.min(z), (np.min(z) + np.max(z)) / 2, np.max(z)])

        a, b, d = optimize_obj_func(x1, x2, y, alpha, a, b, d)
        J_mat[ind1, ind2], _ = obj_func(x1, x2, y, alpha, a, b, d)

        print(":", J_mat[ind1, ind2])

np.set_printoptions(linewidth=300)
print(J_mat)
"""


#ind1 = 0, ind2 = 12
#ind1 = 1, ind2 = 19
#ind1 = 2, ind2 = 6
#ind1 = 3, ind2 = 7
#ind1 = 4, ind2 = 20
#ind1 = 5, ind2 = 22 (23)

#ind1 = 24, ind2 = 21
#ind1 = 25, ind2 = 20

"""
alpha = 0.3
[0.         0.         0.         0.         0.         0.         0.83638497 0.82954393 0.80628197 0.85838709 0.8061533  0.84494671 0.7820894  0.85633189 0.85120281 0.86044876 0.82795762 0.85602357 0.85154513 0.85534536 0.85906523 0.85144828 0.82388269 0.81390641]
 [0.         0.         0.         0.         0.         0.         0.83687945 0.84772062 0.83425963 0.81468114 0.84559872 0.84076897 0.80495283 0.83606218 0.83742747 0.83980674 0.8381982  0.82408454 0.83838252 0.80409929 0.83846657 0.83801553 0.84149472 0.84082679]
 [0.         0.         0.         0.         0.         0.         0.81209465 0.91530241 0.91730807 0.82454227 0.88960823 0.81830232 0.9020144  0.91428983 0.91759474 0.88453089 0.84559227 0.91129338 0.89276875 0.86490968 0.87028274 0.89653819 0.88601097 0.86981593]
 [0.         0.         0.         0.         0.         0.         0.73854847 0.68248447 0.76718877 0.76569357 0.75579045 0.73860461 0.77634208 0.79921907 0.79565503 0.81087809 0.76475183 0.7625826  0.72092859 0.76522235 0.78231834 0.74585072 0.72965857 0.76854328]
 [0.         0.         0.         0.         0.         0.         0.77724401 0.7786081  0.76617921 0.83294186 0.77980711 0.82237073 0.81894874 0.81442404 0.7802412  0.79902519 0.78803976 0.77915395 0.79618149 0.81411709 0.75893466 0.82513399 0.82935166 0.79235974]
 [0.         0.         0.         0.         0.         0.         0.84050945 0.84294682 0.78065722 0.80034087 0.86756146 0.81923701 0.76721665 0.82445669 0.77927997 0.78760323 0.85386953 0.82722201 0.8270252  0.8187785  0.85901432 0.80754004 0.75356301 0.76472418]
 
alpha = 0.25
[0.         0.         0.         0.         0.         0.         0.81364068 0.8183349  0.79343289 0.83507902 0.78475809 0.838002   0.75671144 0.83574753 0.8307229  0.83885697 0.80967309 0.83367606 0.83150659 0.81272456 0.82938662 0.83472338 0.80018711 0.75796279]
 [0.         0.         0.         0.         0.         0.         0.8090361  0.81588625 0.80708137 0.80909971 0.81231106 0.81303301 0.77817068 0.80890027 0.82631122 0.812508   0.8081184  0.79148198 0.81172574 0.80116905 0.81247235 0.80833649 0.81238459 0.8309696 ]
 [0.         0.         0.         0.         0.         0.         0.77802813 0.8862121  0.87051098 0.79978118 0.86129688 0.79085307 0.87517086 0.88963365 0.88884505 0.85584076 0.82244144 0.85660803 0.86492665 0.87266326 0.89251054 0.8632138  0.8564562  0.83539954]
 [0.         0.         0.         0.         0.         0.         0.71497089 0.65773926 0.71652218 0.74209504 0.73161742 0.7135082  0.74089991 0.76896839 0.77416773 0.78598729 0.74486562 0.7450069  0.72198556 0.74052551 0.76277083 0.7148853  0.70565707 0.74454016]
 [0.         0.         0.         0.         0.         0.         0.74441901 0.74628669 0.72747628 0.79648778 0.74743934 0.798028   0.78396348 0.78047798 0.74786668 0.76594212 0.75366348 0.74683371 0.73235485 0.79585682 0.72336176 0.79856105 0.79542052 0.75980457]
 [0.         0.         0.         0.         0.         0.         0.82497668 0.82599901 0.75727813 0.78229506 0.79868875 0.80581476 0.75843892 0.81512657 0.74963015 0.7808519  0.83738264 0.799392   0.77401632 0.80549587 0.84749643 0.79868561 0.7702928  0.75556662]
"""
