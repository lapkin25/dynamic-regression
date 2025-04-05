clear all
format short

# считываем данные из файла
data = dlmread("winter.csv", ";");
[rows, cols] = size(data);
#cols_names = data(1, 2:cols);  # годы
data = data(2:rows, 2:cols);
rows_names = {};  # названия рядов
# открываем файл для считывания названий
fid = fopen("winter.csv");
# читаем заголовок
s = fgetl(fid);
for i = 1:rows-1
  s = fgetl(fid);
  ind = strfind(s, ";")(1);
  rows_names{i} = s(1:ind-1);
  # TODO: посчитать кол-во пустых полей в конце каждой строки
endfor
fclose(fid);

years = 1974 : 2023;
cols = length(years)
data = data(:, 1:cols)';
years_num = length(years);  # длина каждого временного ряда
series_num = rows - 1;  # количество временных рядов
# теперь в каждой строке - данные по каждому году из диапазона years,
#   в каждом столбце - отдельный временной ряд

# R^2 на тестовой выборке
function R2 = test_R2 (x, y, coeff_a, coeff_b, coeff_c)
  n = length(x);
  R2 = 1 - (1 / (n - 1) * sum((x(2:n) - (coeff_a * x(1:n-1) + coeff_b * y(1:n-1) + coeff_c)) .^ 2)) ...
    / (1 / n * sum((x - mean(x)) .^ 2));
endfunction

# R^2 на тестовой выборке
function R2 = test_R2_t (x, y, coeff_alpha, coeff_a0, coeff_ainit, coeff_b, coeff_c)
  n = length(x);
  a = coeff_ainit;
  s = 0;
  for t = 1:n-1
    s = s + (x(t + 1) - (a * x(t) + coeff_b * y(t) + coeff_c)) ^ 2;
    a = coeff_alpha * a * x(t) + coeff_a0;
  endfor
  R2 = 1 - (1 / (n - 1) * s) / (1 / n * sum((x - mean(x)) .^ 2));
endfunction

function [R2, coeff_a, coeff_b, coeff_c] = dynamic_model (x, y)
  n = length(x);  # длина временного ряда
  A = zeros(3);  # матрица СЛАУ
  b = zeros(3, 1);  # правая часть СЛАУ
  A(1, 1) = sum(x(1:n-1) .^ 2);
  A(1, 2) = sum(x(1:n-1) .* y(1:n-1));
  A(1, 3) = sum(x(1:n-1));
  b(1) = sum(x(1:n-1) .* x(2:n));
  A(2, 1) = sum(x(1:n-1) .* y(1:n-1));
  A(2, 2) = sum(y(1:n-1) .^ 2);
  A(2, 3) = sum(y(1:n-1));
  b(2) = sum(x(2:n) .* y(1:n-1));
  A(3, 1) = sum(x(1:n-1));
  A(3, 2) = sum(y(1:n-1));
  A(3, 3) = n-1;
  b(3) = sum(x(2:n));
  coeff = A^(-1) * b;
  coeff_a = coeff(1);
  coeff_b = coeff(2);
  coeff_c = coeff(3);
  R2 = 1 - (1 / (n - 1) * sum((x(2:n) - (coeff_a * x(1:n-1) + coeff_b * y(1:n-1) + coeff_c)) .^ 2)) ...
    / (1 / n * sum((x - mean(x)) .^ 2));
  #R2 = 1 - (sum((x(2:n) - (coeff_a * x(1:n-1) + coeff_b * y(1:n-1) + coeff_c)) .^ 2)) ...
  #  / (sum((x - mean(x)) .^ 2));
endfunction

# p = [alpha, a0, ainit, b, c]
function obj = opt_f (p)
  global xx
  global yy
  alpha = p(1);
  a0 = p(2);
  ainit = p(3);
  b = p(4);
  c = p(5);
  n = length(xx);
  a = ainit;
  obj = 0;
  for t = 1:n-1
    obj = obj + (xx(t + 1) - (a * xx(t) + b * yy(t) + c)) ^ 2;
    #a = alpha * (a - 1) * xx(t) + a0;
    a = alpha * a * xx(t) + a0;
  endfor
  lam = 0.1;
  obj = obj + lam * alpha ^ 2;
endfunction

function [R2, coeff_alpha, coeff_a0, coeff_ainit, coeff_b, coeff_c] = dynamic_model_t (x, y)
  global xx
  global yy
  xx = x;
  yy = y;
  [dummy, a, b, c] = dynamic_model(x, y);  # начальное приближение
  guess = [0, a, a, b, c];
  #[p, obj, info, iter, nf, lambda] = sqp(guess, @opt_f);
  [p, fval, info] = fminsearch(@opt_f, guess);
  coeff_alpha = p(1);
  coeff_a0 = p(2);
  coeff_ainit = p(3);
  coeff_b = p(4);
  coeff_c = p(5);

  a0 = coeff_a0;
  b = coeff_b;
  c = coeff_c;
  alpha = coeff_alpha;
  ainit = coeff_ainit;
  a = ainit;
  s = 0;
  n = length(xx);
  for t = 1:n-1
    s = s + (xx(t + 1) - (a * xx(t) + b * yy(t) + c)) ^ 2;
    #a = alpha * (a - 1) * xx(t) + a0;
    a = alpha * a * xx(t) + a0;
  endfor
  R2 = 1 - (1 / (n - 1) * s) / (1 / n * sum((x - mean(x)) .^ 2));
endfunction

function [R2, coeff_a, coeff_b1, coeff_b2, coeff_c] = dynamic_model_3 (x, y, z)
  n = length(x);  # длина временного ряда
  A = zeros(4);  # матрица СЛАУ
  b = zeros(4, 1);  # правая часть СЛАУ
  A(1, 1) = sum(x(1:n-1) .^ 2);
  A(1, 2) = sum(x(1:n-1) .* y(1:n-1));
  A(1, 3) = sum(x(1:n-1) .* z(1:n-1));
  A(1, 4) = sum(x(1:n-1));
  b(1) = sum(x(1:n-1) .* x(2:n));
  A(2, 1) = sum(x(1:n-1) .* y(1:n-1));
  A(2, 2) = sum(y(1:n-1) .^ 2);
  A(2, 3) = sum(z(1:n-1) .* y(1:n-1));
  A(2, 4) = sum(y(1:n-1));
  b(2) = sum(x(2:n) .* y(1:n-1));
  A(3, 1) = sum(x(1:n-1) .* z(1:n-1));
  A(3, 2) = sum(z(1:n-1) .* y(1:n-1));
  A(3, 3) = sum(z(1:n-1) .^ 2);
  A(3, 4) = sum(z(1:n-1));
  b(3) = sum(x(2:n) .* z(1:n-1));
  A(4, 1) = sum(x(1:n-1));
  A(4, 2) = sum(y(1:n-1));
  A(4, 3) = sum(z(1:n-1));
  A(4, 4) = n-1;
  b(4) = sum(x(2:n));
  coeff = A^(-1) * b;
  coeff_a = coeff(1);
  coeff_b1 = coeff(2);
  coeff_b2 = coeff(3);
  coeff_c = coeff(4);
  R2 = 1 - (1 / (n - 1) * sum((x(2:n) - (coeff_a * x(1:n-1) + coeff_b1 * y(1:n-1) + coeff_b2 * z(1:n-1) + coeff_c)) .^ 2)) ...
    / (1 / n * sum((x - mean(x)) .^ 2));
endfunction

function R2 = dynamic_coef (data)
  n = columns(data);  # число рядов с данными
  data_size = rows(data);  # длина рядов
  R2 = zeros(n);
  for i = 1:n
    for j = 1:n
      if (i == j)
        R2(i, j) = 0;
      else
        [R2(i, j), a, b, c] = dynamic_model(data(:, i), data(:, j));
      endif
    endfor
  endfor
endfunction

function R2 = dynamic_coef_t (data)
  n = columns(data);  # число рядов с данными
  data_size = rows(data);  # длина рядов
  R2 = zeros(n);
  for i = 1:n
    for j = 1:n
      if (i == j)
        R2(i, j) = 0;
      else
        [R2(i, j), coeff_alpha, coeff_a0, coeff_ainit, coeff_b, coeff_c] = dynamic_model_t(data(:, i), data(:, j));
      endif
    endfor
  endfor
endfunction

function R2 = test_dynamic_coef (data, test_data)
  n = columns(data);  # число рядов с данными
  data_size = rows(data);  # длина рядов
  R2 = zeros(n);
  for i = 1:n
    for j = 1:n
      if (i == j)
        R2(i, j) = 0;
      else
        [R2_, a, b, c] = dynamic_model(data(:, i), data(:, j));
        R2(i, j) = test_R2(test_data(:, i), test_data(:, j), a, b, c);
      endif
    endfor
  endfor
endfunction

function R2 = test_dynamic_coef_t (data, test_data)
  n = columns(data);  # число рядов с данными
  data_size = rows(data);  # длина рядов
  R2 = zeros(n);
  for i = 1:n
    for j = 1:n
      if (i == j)
        R2(i, j) = 0;
      else
        [R2_, alpha, a0, ainit, b, c] = dynamic_model_t(data(:, i), data(:, j));
        xx = data(:, i);
        yy = data(:, j);
        a = ainit;
        for t = 1:length(xx)  # без -1
          a = alpha * a * xx(t) + a0;
        endfor
        ainit_new = a;
        R2(i, j) = test_R2_t(test_data(:, i), test_data(:, j), alpha, a0, ainit_new, b, c);
      endif
    endfor
  endfor
endfunction

function R2 = dynamic_coef_3 (data)
  n = columns(data);  # число рядов с данными
  data_size = rows(data);  # длина рядов
  R2 = zeros(n);
  for i = 1:n
    for j = 1:n
      if (i == j)
        R2(i, j) = 0;
      else
        R2(i, j) = 0;
        for k = 1:n
          if (k != i && k != j)
            [R2_, a, b1, b2, c] = dynamic_model_3(data(:, i), data(:, j), data(:, k));
            R2(i, j) = max(R2(i, j), R2_);
          endif
        endfor
      endif
    endfor
  endfor
endfunction

#years_range = 43:50;

start_year_ind = 42;  # 36;
years_range_len = 8;  #10;
test_years_range_len = 1;

years_range = start_year_ind : start_year_ind + years_range_len - 1;
test_years_range = start_year_ind + years_range_len - 1 : start_year_ind + years_range_len + test_years_range_len - 1;

#years_range = 42:50;

src_data = data;
src_years = years;

data1 = data(years_range, :);
years1 = years(years_range);
test_years = years(test_years_range)
test_data = data(test_years_range, :);
years = years1
data = data1

fixed_point_format(true)


#{
# корреляционная матрица
r = corrcoef(data)
async_r = corr(data(2:length(years), :), data(1:length(years)-1, :))
R2 = dynamic_coef(data)
# R2_ = dynamic_coef_3(data)
#R2_test = test_dynamic_coef(data, test_data)
R2_t = dynamic_coef_t(data)
#}


#R2_13 = R2(:, 13)




i = 1  #3
j = 22  #8
#kk = 17 # 4 или 13
x = data(:, i);
y = data(:, j);
test_x = test_data(:, i);
test_y = test_data(:, j);





# три ряда
#{

R2_ = zeros(series_num, 1);
for k = 1:series_num
  z = data(:, k);
  if (k != i && k != j)
    [R2_(k), a, b1, b2, c] = dynamic_model_3(x, y, z);
  endif
endfor
R2_

#}


#corr(x(2:length(years)), y(1:length(years)-1))

[R2, a, b, c] = dynamic_model(x, y)
xx(1) = x(1);
for t = 1:length(x)-1
  xx(t + 1) = a * x(t) + b * y(t) + c;
endfor
figure
plot(years, xx, years, x)
#figure
#plot(years, y)



[R2, alpha, a0, ainit, b, c] = dynamic_model_t(x, y)
#r_ = r(i, j)
#r_a = async_r(i, j)

xx(1) = x(1);
a = ainit;
for t = 1:length(x)-1
  xx(t + 1) = a * x(t) + b * y(t) + c;
  #a = alpha * (a - 1) * x(t) + a0;
  a = alpha * a * x(t) + a0;
endfor
#figure
#plot(years, xx, years, x)
#figure
#plot(years, y)

#!!!
#{
t = length(x);
#tt = length(test_x);
xx(t + 1) = a * x(t) + b * y(t) + c;
xx
x'
y'
#figure
#plot([years(2:length(years)) 2024], xx(2:length(xx)), years, x, years, y, '.-')
plot([years(2:length(years)) years(length(years)) + 1], xx(2:length(xx)), years, x, '.-')
#figure
#plot(years, y, '.-')
#}


tt = length(test_x);
test_xx(1) = test_x(1);
# крайнее значение a - уже вычислено
for t = 1:length(test_x)-1
  test_xx(t + 1) = a * test_x(t) + b * test_y(t) + c;
  #a = alpha * (a - 1) * test_x(t) + a0;
  a = alpha * a * test_x(t) + a0;
endfor
test_xx(tt + 1) = a * test_x(tt) + b * test_y(tt) + c;
figure
plot(years, xx, years, x)
#figure
hold on
plot([test_years test_years(length(test_years)) + 1], [xx(length(xx)) test_xx(2:length(test_xx))], 'b.--', test_years, test_x, 'r.--')









# три ряда
#{

k = kk
z = data(:, k);
[R2, a, b1, b2, c] = dynamic_model_3(x, y, z)

xx(1) = x(1);
for t = 1:length(x)-1
  xx(t + 1) = a * x(t) + b1 * y(t) + b2 * z(t) + c;
endfor
t = length(x);
xx(t + 1) = a * x(t) + b1 * y(t) + b2 * z(t) + c;
figure
plot([years(2:length(years)) years(length(years)) + 1], xx(2:length(xx)), years, x, '.-')
figure
plot(years, y, '.-')
figure
plot(years, z, '.-')

#}


#{
# первый этап вычислений
cnt = 1;
for start_year_ind = 37:41
  years_range_len = 10;  #8;  #10;
  #test_years_range_len = 5;

  years_range = start_year_ind : start_year_ind + years_range_len - 1;
  test_years_range = start_year_ind + years_range_len : start_year_ind + years_range_len + test_years_range_len - 1;

  data = src_data(years_range, :);
  years = src_years(years_range);
  #test_data = src_data(test_years_range, :);
  #test_years = src_years(test_years_range);

  # корреляционная матрица
  R2_stat{cnt} = dynamic_coef(data);
  R2_t_stat{cnt} = dynamic_coef_t(data);
  #R2_test_stat{cnt} = test_dynamic_coef(data, test_data)
  cnt += 1;
endfor

#R2_mean = (R2_stat{1} + R2_stat{2} + R2_stat{3}) / 3;
#R2_test_mean = (R2_test_stat{1} + R2_test_stat{2} + R2_test_stat{3}) / 3;
#R2_test_mean(R2_test_mean < 0) = 0

R2_mean = (R2_stat{1} + R2_stat{2} + R2_stat{3} + R2_stat{4} + R2_stat{5}) / 5
R2_t_mean = (R2_t_stat{1} + R2_t_stat{2} + R2_t_stat{3} + R2_t_stat{4} + R2_t_stat{5}) / 5
#}



# второй этап вычислений
cnt = 1;
for start_year_ind = 36:38
  years_range_len = 10;
  test_years_range_len = 3;

  years_range = start_year_ind : start_year_ind + years_range_len - 1;
  test_years_range = start_year_ind + years_range_len : start_year_ind + years_range_len + test_years_range_len - 1;

  data = src_data(years_range, :);
  years = src_years(years_range);
  test_data = src_data(test_years_range, :);
  test_years = src_years(test_years_range);

  # корреляционная матрица
  R2_stat{cnt} = dynamic_coef(data);
  R2_t_stat{cnt} = dynamic_coef_t(data);
  R2_test_stat{cnt} = test_dynamic_coef(data, test_data);
  R2_t_test_stat{cnt} = test_dynamic_coef_t(data, test_data);
  cnt += 1;
endfor

#R2_mean = (R2_stat{1} + R2_stat{2} + R2_stat{3}) / 3;
#R2_test_mean = (R2_test_stat{1} + R2_test_stat{2} + R2_test_stat{3}) / 3;
#R2_test_mean(R2_test_mean < 0) = 0

#R2_mean = (R2_stat{1} + R2_stat{2} + R2_stat{3} + R2_stat{4} + R2_stat{5}) / 5
#R2_test_mean = (R2_test_stat{1} + R2_test_stat{2} + R2_test_stat{3} + R2_test_stat{4} + R2_test_stat{5}) / 5;
R2_mean = (R2_stat{1} + R2_stat{2} + R2_stat{3}) / 3
R2_t_mean = (R2_t_stat{1} + R2_t_stat{2} + R2_t_stat{3}) / 3
R2_test_mean = (R2_test_stat{1} + R2_test_stat{2} + R2_test_stat{3}) / 3;
R2_test_mean(R2_test_mean < 0) = 0

