import math
import statistics
from scipy.stats import chi2
from scipy.stats import t
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

print("ЗАДАНИЕ 1.", "\nВ результате эксперимента получена выборка из 100 чисел: ")

experimental_values = [
    72, 101, 65, 64, 35, 96, 67, 30, 93, 123,
    16, 135, 138, 90, 158, 121, 49, 137, 89, 145,
    68, 150, 88, 93, 53, 38, 159, 40, 76, 37,
    104, 34, 99, 102, 78, 128, 124, 52, 98, 139,
    18, 81, 25, 115, 71, 94, 84, 55, 131, 70,
    87, 126, 57, 141, 15, 125, 149, 36, 103, 82,
    39, 140, 77, 54, 100, 86, 129, 48, 80, 144,
    69, 109, 130, 147, 146, 73, 105, 113, 17, 94,
    21, 97, 51, 50, 19, 142, 32, 66, 110, 114,
    92, 33, 112, 91, 61, 85, 71, 151, 56, 41
]
values_array = np.array(experimental_values).reshape(10, 10)
df = pd.DataFrame(values_array)
print(df.to_string(index=False, header=False))

experimental_values.sort()

print("1. Записываем числовые значения (варианты) в порядке возрастания, получим вариационный ряд: ")
number = 0

values_array = np.array(experimental_values).reshape(10, 10)
df = pd.DataFrame(values_array)
print(df.to_string(index=False, header=False))

scope_of_variation = experimental_values[99] - experimental_values[0]

print("2. Находим размах вариации: \nxₘₐₓ - xₘᵢₙ = ", experimental_values[99], " - ", experimental_values[0], " = ",
      round(scope_of_variation, 4), end="\n")

optimal_number_of_intervals_k = round(1 + 3.322 * math.log10(100))

print("оптимальное число интервалов: \nk = 1 + 3.322lg(n) = 1 + 3.322lg(100) = ", 1 + 3.322 * math.log10(100), " ≈ ",
      optimal_number_of_intervals_k, end="\n")

partial_interval_length_h = scope_of_variation / optimal_number_of_intervals_k

print("и длину частичного интервала: \nh = (xₘₐₓ - xₘᵢₙ)/8 = ", round(scope_of_variation, 4), "/", 8, " = ",
      round(partial_interval_length_h, 4), end="\n")

interval_boundaries = [experimental_values[0]]

while number < optimal_number_of_intervals_k:
    interval_boundaries.append(interval_boundaries[number] + partial_interval_length_h)
    number += 1
else:
    number = 0

print("Выпишем границы интервалов")
df = pd.DataFrame(interval_boundaries, columns=["aᵢ"])
print(df)

print("Подсчитаем число вариант, попавших в каждый интервал, т.е. находим частоты mᵢ , запишем интервальное,",
      "\nраспределение частот выборки:")

distribution_of_sampling_frequencies_m = [0]
results = []

for num in experimental_values:
    if (num >= interval_boundaries[number]) and (num <= interval_boundaries[number + 1]):
        distribution_of_sampling_frequencies_m[number] += 1
    else:
        distribution_of_sampling_frequencies_m.append(1)
        results.append([
            f"{round(interval_boundaries[number], 4)}-{round(interval_boundaries[number + 1], 4)}",
            round(distribution_of_sampling_frequencies_m[number], 4)
        ])
        number += 1
else:
    results.append([
        f"{round(interval_boundaries[number], 4)} - {round(interval_boundaries[number + 1], 4)}",
        round(distribution_of_sampling_frequencies_m[number], 4)
    ])
    number = 0

df = pd.DataFrame(results, columns=["Интервалы", "mᵢ"])
print(df)

number_of_experimental_values_n = len(experimental_values)

print("3. Находим относительные частоты: \nwᵢ = mᵢ/n, n = ", number_of_experimental_values_n,
      "\nи их плотности \nwᵢ/h, h = ", round(partial_interval_length_h, 4))

relative_frequencies_w = []
densities = []

results = []

for num in distribution_of_sampling_frequencies_m:
    relative_frequencies_w.append(num / number_of_experimental_values_n)
    densities.append(relative_frequencies_w[number] / partial_interval_length_h)
    results.append([
        f"{round(interval_boundaries[number], 4)} - {round(interval_boundaries[number + 1], 4)}",
        round(num, 4),
        round(relative_frequencies_w[number], 4),
        round(densities[number], 4)
    ])
    number += 1
else:
    number = 0

df = pd.DataFrame(results, columns=["Интервалы", "mᵢ", "wᵢ", "wᵢ/h"])
print(df)

print("Строим гистограмму относительных частот (масштаб на осях разный).\n(график на экране)")

fig, ax = plt.subplots()
ax.hist(interval_boundaries[:-1], weights=densities, bins=interval_boundaries)

ax.set_title("Гистограмма относительных частот")
ax.set_xlabel("Интервалы")
ax.set_ylabel("wᵢ/h")

plt.show()

print("Находим значения эмпирической функции:", "\nF(x)=nₓ/n, где nₓ - накопленная частота.",
      "\nВ качестве аргумента функции рассматриваем концы интервалов:")

values_of_the_empirical_function_F = [0]
print(f"F({round(interval_boundaries[0], 4)}) = {round(values_of_the_empirical_function_F[0], 4)}/{100} = {0}")

buff_num = 0
for num in distribution_of_sampling_frequencies_m:
    values_of_the_empirical_function_F.append((buff_num + num) / 100)
    print(f"F({round(interval_boundaries[number + 1], 4)}) = ({round(buff_num, 4)} + {round(num, 4)})/{100} =",
          f"{round(values_of_the_empirical_function_F[number + 1], 4)}")
    buff_num += num
    number += 1
else:
    number = 0
    buff_num = 0

print("Строим график эмпирической функции или кумулятивной кривой выборки:\n(график на экране)\n")

plt.clf()
plt.plot(interval_boundaries, values_of_the_empirical_function_F, marker='.')
plt.xlim(min(interval_boundaries), max(interval_boundaries))
plt.ylim(min(values_of_the_empirical_function_F), max(values_of_the_empirical_function_F))

plt.title("График эмпирической функции")
plt.xlabel("Интервалы")
plt.ylabel("F(x)")

plt.show()

print("4. Находим выборочное среднее, среднее по квадратам, выборочную дисперсию,\nсреднее квадратическое отклонение, ",
      "исправленную выборочную дисперсию,\nисправленное среднее квадратическое отклонение.",
      "\nСоставим расчетную таблицу.")

xi_mi = 0
xi2_mi = 0
interval_boundaries_average = []

results = []

while number < len(interval_boundaries) - 1:
    interval_boundaries_average.append((interval_boundaries[number] + interval_boundaries[number + 1]) / 2)
    xi_mi += ((interval_boundaries[number] + interval_boundaries[number + 1]) / 2) \
             * distribution_of_sampling_frequencies_m[number]
    xi2_mi += (((interval_boundaries[number] + interval_boundaries[number + 1]) / 2)
               ** 2) * distribution_of_sampling_frequencies_m[number]
    results.append([
        f"{round(interval_boundaries[number], 4)} - {round(interval_boundaries[number + 1], 4)}",
        round((interval_boundaries[number] + interval_boundaries[number + 1]) / 2, 4),
        round(distribution_of_sampling_frequencies_m[number], 4),
        round(((interval_boundaries[number] + interval_boundaries[number + 1]) / 2)
              * distribution_of_sampling_frequencies_m[number], 4),
        round((((interval_boundaries[number] + interval_boundaries[number + 1]) / 2)
               ** 2) * distribution_of_sampling_frequencies_m[number], 4)
    ])
    number += 1
else:
    number = 0

df = pd.DataFrame(results, columns=["aᵢ - aᵢ₊₁", "xᵢ", "mᵢ", "xᵢ•mᵢ", "xᵢ²•mᵢ"])
print(df)

print(f"Сумма m = 100 x•m = {round(xi_mi, 4)} x^2•m = {round(xi2_mi, 4)}")

sample_mean_x = xi_mi / 100
squared_mean_x2 = xi2_mi / 100
sample_variance_D = squared_mean_x2 - (sample_mean_x ** 2)
mean_square_deviation_G = math.sqrt(sample_variance_D)
corrected_sample_variance_s2 = (100 / 99) * sample_variance_D
corrected_mean_square_deviation_s = math.sqrt(corrected_sample_variance_s2)

print(f"xᵦ = {round(xi_mi, 4)}/100 = {round(sample_mean_x, 4)}")
print(f"x² = {round(xi2_mi, 4)}/100 = {round(squared_mean_x2, 4)}")
print(f"Dᵦ = {round(squared_mean_x2, 4)}-{round(sample_mean_x, 4)}^2 = {round(sample_variance_D, 4)}")
print(f"σᵦ = {round(sample_variance_D, 4)}^(1/2) = {round(mean_square_deviation_G, 4)}")
print(f"s² = (100/99)•{round(sample_variance_D, 4)} = {round(corrected_sample_variance_s2, 4)}")
print(f"s = {round(corrected_sample_variance_s2, 4)}^(1/2) = {round(corrected_mean_square_deviation_s, 4)}\n")

print("5. По виду гистограммы выдвигаем гипотезу о нормальном распределении",
      "\nв генеральной совокупности признака Х, причем",
      f"\nM(X) = a = xᵦ = {round(sample_mean_x, 4)}; s = σ(X) = {round(corrected_mean_square_deviation_s, 4)}.",
      "\nНаходим",
      f"\nxᵦ - 3•s = {round(sample_mean_x, 4)} - 3•{round(corrected_mean_square_deviation_s, 4)} =",
      f"{round(sample_mean_x-(3*corrected_mean_square_deviation_s), 4)}",
      f"\nxᵦ + 3•s = {round(sample_mean_x, 4)} + 3•{round(corrected_mean_square_deviation_s, 4)} =",
      f"{round(sample_mean_x+(3*corrected_mean_square_deviation_s), 4)}",
      f"\nВыборка от  xₘᵢₙ = {round(experimental_values[0], 4)} до xₘₐₓ =",
      f"{round(experimental_values[-1], 4)} входит в интервал",
      "\n(xᵦ - 3•s; xᵦ + 3•s),",
      "\nт.е. имеет место «правило трех сигм» для этого распределения.",
      "\nПо критерию Пирсона надо сравнивать эмпирические и",
      "\nтеоретические частоты вариант. Эмпирические частоты mi даны.",
      "\nТеоретические частоты mᵢ' найдем по формуле",
      "\nmᵢ'= n•P = 100•P(aᵢ<X<aᵢ₊₁) = 100(Ф((aᵢ₊₁ - x)/s) - Ф((aᵢ - x)/s)).",
      "\nСоставим вспомогательную таблицу:")

number = 1
buff_num = 0
bukvi_FF = []
P = []
results = []

for num in interval_boundaries:
    bukvi_FF.append(norm.cdf((num - sample_mean_x) / corrected_mean_square_deviation_s) - 0.5)
    if num == interval_boundaries[-1]:
        results.append([
            round(num, 4),
            round((num - sample_mean_x) / corrected_mean_square_deviation_s, 4),
            round(bukvi_FF[number - 1], 4),
            round(buff_num, 4)
        ])
    else:
        results.append([
            round(num, 4),
            round((num - sample_mean_x) / corrected_mean_square_deviation_s, 4),
            round(bukvi_FF[number - 1], 4),
            round((norm.cdf((interval_boundaries[number] - sample_mean_x) / corrected_mean_square_deviation_s) - 0.5)
                  - (norm.cdf((num - sample_mean_x) / corrected_mean_square_deviation_s) - 0.5), 4)
        ])
        buff_num += (norm.cdf((interval_boundaries[number] - sample_mean_x) / corrected_mean_square_deviation_s)
                     - 0.5) - (norm.cdf((num - sample_mean_x) / corrected_mean_square_deviation_s) - 0.5)
        P.append((norm.cdf((interval_boundaries[number] - sample_mean_x) / corrected_mean_square_deviation_s)
                  - 0.5) - (norm.cdf((num - sample_mean_x) / corrected_mean_square_deviation_s) - 0.5))
    number += 1
else:
    number = 0

df = pd.DataFrame(results, columns=["aᵢ", "(aᵢ-x)/s", "Ф(uᵢ)", "Pᵢ"])
print(df)

print("Статистика имеет распределение «хи-квадрат» лишь при n → ∞, поэтому необходимо, чтобы в каждом",
      "\nинтервале было не менее 5 значений. Если mᵢ < 5, имеет смысл объединить соседние интервалы.")

n_P = []
buff_num = 0
m_mines_m = []
m_mines_m2 = []
m_mines_m2_divide_m = []
buff_num1 = 0
buff_num2 = 0


for num in distribution_of_sampling_frequencies_m:
    n_P.append(100 * P[number])
    m_mines_m.append(num - n_P[number])
    number += 1
else:
    number = len(distribution_of_sampling_frequencies_m) - 1

while number >= 0:
    if distribution_of_sampling_frequencies_m[number] < 5:
        buff_num1 += m_mines_m[number]
        buff_num2 += n_P[number]
    else:
        buff_num1 += m_mines_m[number]
        buff_num2 += n_P[number]
        m_mines_m2.append(buff_num1 ** 2)
        m_mines_m2_divide_m.append((buff_num1 ** 2) / buff_num2)
        buff_num1 = 0
        buff_num2 = 0
    number -= 1
else:
    number = 0

m_mines_m2.reverse()
m_mines_m2_divide_m.reverse()
index = 0
results = []

for num in distribution_of_sampling_frequencies_m:
    if distribution_of_sampling_frequencies_m[number] < 5:
        results.append([
            round(num, 4),
            round(n_P[number], 4),
            round(m_mines_m[number], 4),
            "",
            ""
        ])
    else:
        results.append([
            round(num, 4),
            round(n_P[number], 4),
            round(m_mines_m[number], 4),
            round(m_mines_m2[index], 4),
            round(m_mines_m2_divide_m[index], 4)
        ])
        index += 1
    number += 1
else:
    number = 0

df = pd.DataFrame(results, columns=["mᵢ", "100•Pᵢ=mᵢ'", "mᵢ-mᵢ'", "(mᵢ-mᵢ')²", "((mᵢ-mᵢ')²)/mᵢ'"])
print(df)

print(f"m(сумма)' = {round(sum(n_P), 4)} x²_набл = {round(sum(m_mines_m2_divide_m), 4)}")

print(f"Находим X²_крит(a, k = l - 3) = X²_крит(0.05; {len(m_mines_m2)} - 3) = X²_крит(0.05; {len(m_mines_m2) - 3}) =",
      round(chi2.ppf(1-.05, len(m_mines_m2) - 3), 4))

print(f"Так как x²_набл = {round(sum(m_mines_m2_divide_m), 4)} < X²_крит, то гипотеза H₀ о нормальном",
      "\nраспределении принимается.",
      "\nПо критерию Колмогорова надо сравнить",
      f"\nλ_опыт = n¹⁄²•max|F*(xᵢ) - F(xᵢ)| и λ_крит(0.05) = 1.358.",
      "\nДля нормального распределения \nF(xᵢ) = 0.5 + Ф((xᵢ - x)/s).",
      f"\nВ качестве xᵢ возьмем aᵢ (i = 0.{len(interval_boundaries) - 1}).",
      "\nСоставим таблицу:")

module_f_minus_f = []
results = []

for a in interval_boundaries:
    module_f_minus_f.append(abs((values_of_the_empirical_function_F[number]) - (0.5 + bukvi_FF[number])))
    results.append([
        round(a, 4),
        round(values_of_the_empirical_function_F[number], 4),
        f"0.5+({round(bukvi_FF[number], 4)})",
        round(0.5 + bukvi_FF[number], 4),
        round(abs(module_f_minus_f[number]), 4)
    ])
    number += 1
else:
    number = 0

df = pd.DataFrame(results, columns=["aᵢ", "F*(aᵢ)", "0.5+Ф(uᵢ)", "F(aᵢ)", "|F*(aᵢ)-F(aᵢ)|"])
print(df)

print(f"max|F*(aᵢ) - F(aᵢ)| = {round(max(module_f_minus_f), 4)}",
      f"\nλ_опыт = 100¹⁄²•{round(max(module_f_minus_f), 4)} = {round((max(module_f_minus_f) * 10), 4)}",
      "< λ_крит = 1.358.",
      "\nГипотеза Н₀ о нормальном распределении не отвергается.\n")

print("6. Плотность нормального распределения:",
      f"f(x) = 1 / ({round(corrected_mean_square_deviation_s, 4)}•(2•{round(math.pi, 4)})½)",
      f"• e^-((x - {round(sample_mean_x, 4)})²)/(2•{round(corrected_mean_square_deviation_s, 4)}²))")

density_of_the_normal_distribution_f = []
results = []

for i in interval_boundaries_average:
    density_of_the_normal_distribution_f.append(1 / (corrected_mean_square_deviation_s * ((2 * math.pi) ** 0.5))
                                                * math.e ** -(((i - sample_mean_x) ** 2)
                                                              / (2 * (corrected_mean_square_deviation_s ** 2))))
    results.append([
        round(i, 4),
        round(density_of_the_normal_distribution_f[number], 4)
    ])
    number += 1
else:
    number = 0

df = pd.DataFrame(results, columns=["xᵢ", "f(xᵢ)"])
print(df)

print("Откладываем эти пары значений на гистограмме относительных \nчастот, соединяем плавной линией.",
      "\n(график на экране)\n")

plt.clf()
plt.close('all')
fig, ax = plt.subplots()
ax.hist(interval_boundaries[:-1], weights=densities, bins=interval_boundaries)

plt.plot(interval_boundaries_average, density_of_the_normal_distribution_f, marker='.')

plt.title("Гистограмма относительных частот")
plt.xlabel("xᵢ")
plt.ylabel("wᵢ/h")

plt.show()

print("7. Если СВ Х генеральной совокупности распределена нормально, \nто с надежностью y = 0.95 можно уверждать, что",
      "математическое \nожидание a СВ Х покрывается доверительным интервалом (x_ср - б, x_ср + б),",
      "\nгде \nб = (s/n¹⁄²)•tᵧ )- точность оценки.",
      f"\nВ нашей задаче n = 100, s = {round(corrected_mean_square_deviation_s, 4)},",
      "tᵧ = (y, n) = t(0.95, 100) = 1.984.",
      "\nТогда",
      f"б = ({round(corrected_mean_square_deviation_s, 4)} / 10)•1.984 =",
      round((corrected_mean_square_deviation_s / 10) * 1.984, 4), "\nСледовательно",
      f"\nx_сред - б =",
      round(statistics.mean(interval_boundaries), 4), "-", round((corrected_mean_square_deviation_s / 10) * 1.984, 4),
      "=", round(statistics.mean(interval_boundaries) - ((corrected_mean_square_deviation_s / 10) * 1.984), 4),
      f"\nx_сред + б =",
      round(statistics.mean(interval_boundaries), 4), "+", round((corrected_mean_square_deviation_s / 10) * 1.984, 4),
      "=", round(statistics.mean(interval_boundaries) + ((corrected_mean_square_deviation_s / 10) * 1.984), 4),
      "\nТаким образом, доверительный интервал для математического ожидания",
      f"\n({round(statistics.mean(interval_boundaries) - (corrected_mean_square_deviation_s / 10) * 1.984, 4)},"
      f" {round(statistics.mean(interval_boundaries) + (corrected_mean_square_deviation_s / 10) * 1.984, 4)})",
      "\nПри чем",
      f"\nP({round(statistics.mean(interval_boundaries) - (corrected_mean_square_deviation_s / 10) * 1.984, 4)} < a < {round(statistics.mean(interval_boundaries) - (corrected_mean_square_deviation_s / 10) * 1.984, 4)}) = 0.95",
      "\nДоверительный интервал. покрывающий среднее квадратическое отклонение σ с надежностью y = 0.95",
      "\ns(1 - q) < σ < s(1 + q), \nгде q = q(y ,n) = q(0.95; 100) = 0.143",
      "\nТочность оценки", f"\nб = s•q = {round(corrected_mean_square_deviation_s, 4)}•0.143 = {round(corrected_mean_square_deviation_s * 0.143, 4)};",
      "\nСледовательно"
      f"\ns - б = {round(corrected_mean_square_deviation_s, 4)} - {round(corrected_mean_square_deviation_s * 0.143, 4)} =",
      f"{round(corrected_mean_square_deviation_s - (corrected_mean_square_deviation_s * 0.143), 4)}",
      f"\ns + б = {round(corrected_mean_square_deviation_s, 4)} + {round(corrected_mean_square_deviation_s * 0.143, 4)} =",
      f"{round(corrected_mean_square_deviation_s + (corrected_mean_square_deviation_s * 0.143), 4)}",
      f"\nТаким образом, доверительный интервал для среднего квадратического отклонения \nσ ∈",
      f"({round(corrected_mean_square_deviation_s - (corrected_mean_square_deviation_s * 0.143), 4)},"
      f" {round(corrected_mean_square_deviation_s + (corrected_mean_square_deviation_s * 0.143), 4)})",
      f"\nПричем \nP({round(corrected_mean_square_deviation_s - (corrected_mean_square_deviation_s * 0.143), 4)}",
      "< σ <",
      f"{round(corrected_mean_square_deviation_s + (corrected_mean_square_deviation_s * 0.143), 4)}) = 0.95\n")

print("ЗАДАНИЕ 2. \nДано интервальное распределение частот некоторой \nсовокупности относительно признака X :")

a = [0, 54, 108, 162, 216, 270, 324, 378]
m = [51, 26, 21, 8, 6, 5, 3]
h = a[1] - a[0]

results = []

for i in m:
    results.append([
        f"{round(a[number], 4)} - {round(a[number + 1], 4)}",
        round(i, 4)
    ])
    number += 1
else:
    number = 0

df = pd.DataFrame(results, columns=["Интервалы", "mᵢ"])
print(df)

print("\nСоставим таблицу, в которой найдем плотность частоты mᵢ/h , середины интервалов",
      "\nxᵢ, произведения xᵢmᵢ, xᵢ²mᵢ для построения полигона и гистограммы частот",
      f"\nи нахождения числовых характеристик выборки. Длина интервалов h = {h}.")

x = []
m_divide_h = []
x_multiply_m = []
x2_multiply_m = []

results = []

for i in m:
    x.append((a[number] + a[number + 1]) / 2)
    m_divide_h.append(i / h)
    x_multiply_m.append(x[number] * i)
    x2_multiply_m.append((x[number] ** 2) * i)
    results.append([
        f"{round(a[number], 4)} — {round(a[number + 1], 4)}",
        round(x[number], 4),
        round(i, 4),
        round(m_divide_h[number], 4),
        round(x_multiply_m[number], 4),
        round(x2_multiply_m[number], 4)
    ])
    number += 1
else:
    number = 0

df = pd.DataFrame(results, columns=["aᵢ - aᵢ₊₁", "xᵢ", "mᵢ", "mᵢ/h", "xᵢ•mᵢ", "xᵢ²•mᵢ"])
print(df)

n = sum(m)
sum_x_multiply_m = sum(x_multiply_m)
sum_x2_multiply_m = sum(x2_multiply_m)

print(f"Суммы: n = {n}   x•m = {round(sum_x_multiply_m, 4)}   x²•m = {round(sum_x2_multiply_m, 4)}")

x_B = sum_x_multiply_m / n
x2 = sum_x2_multiply_m / n
D_B = x2 - (x_B ** 2)
б_B = D_B ** (1/2)

print(f"xᵦ = {round(sum_x_multiply_m, 4)}/{n} = {round(x_B, 4)}",
      f"\nx² = {round(sum_x2_multiply_m)}/{n} = {round(x2, 4)}",
      f"\nDᵦ = {round(x2, 4)} - {round(x_B, 4)}² = {round(D_B, 4)}",
      f"\nσᵦ = {round(б_B, 4)}")

print(f"Строим эмпирическую функцию распределения F*(aᵢ) = nₐᵢ/n,\naᵢ —",
      f"концы интервалов i = 0.{len(a) - 1}, n = {n} ")

F_a = [0]
buff_num = 0

results = []

results.append([
    round(a[0], 4),
    round(F_a[0], 4)
])

for i in m:
    buff_num += i
    F_a.append(buff_num / n)
    results.append([
        round(a[number + 1], 4),
        f"{round(buff_num, 4)}/{round(n, 4)} = {round(F_a[number + 1], 4)}"
    ])
    number += 1
else:
    number = 0

df = pd.DataFrame(results, columns=["aᵢ", "F*(aᵢ)"])
print(df)

print("(Вывод графиков)")

plt.clf()
plt.close("all")
plt.plot(x, m, marker='.')

plt.xlabel("x")
plt.ylabel("mᵢ")

plt.show()

plt.clf()
plt.close("all")
fig, ax = plt.subplots()
ax.hist(a[:-1], weights=m_divide_h, bins=a)

ax.set_xlabel("x")
ax.set_ylabel("mᵢ/h")

plt.show()

plt.clf()
plt.close("all")
plt.plot(a, F_a, marker='.')

plt.xlabel("x")
plt.ylabel("mᵢ")

plt.show()

lyambda = 1 / x_B

print("По виду полигона частот, гистограммы, F*(x) выдвигаем гипотезу,",
      "\nо показательном распределении признака X в генеральной совокупности.",
      "\nПризнаком этого распределения является совпадение:",
      "\nM(X) = σ(X) = 1/λ.",
      f"\nВ данном случае, xᵦ и σᵦ достаточно близки: λ = 1/xᵦ = {round(lyambda, 4)}.",
      "\nПлотность распределения")

print("        | 0, если x < 0,",
      "\nf(x) = {    ",
      f"\n        | {round(lyambda, 4)}e^(-{round(lyambda, 4)}x), если x >= 0.")

print("Теоретическая функция распределения")

print("        | 0, если x < 0,",
      "\nF(x) = {    ",
      f"\n        | 1 - e^(-{round(lyambda, 4)}x), если x >= 0.")

print("Подтвердим или опровергнем гипотезу Н0: генеральная совокупность признака",
      "\nX подчиняется показательному закону распределения." 
      "\nа) Критерий Пирсона. Находим теоретические (выравнивающие) частоты",
      "\nmᵢ' = nPᵢ = n•P(aᵢ < X < aᵢ₊₁) = n•(e^(-λai) - e^(-λai+1)).",
      "\nСравниваем",
      f"\nX²_крит(a, k = l - 2) = X²_крит(0.05; {len(m_mines_m2_divide_m)} - 2) = X²_крит(0.05;",
      f"{len(m_mines_m2_divide_m) - 2}) =",
      f"{round(chi2.ppf(1-.05, len(m_mines_m2) - 3), 4)}")

P = []
n_P = []
m_mines_m = []
m_mines_m2_divide_m = []

while number < 7:
    P.append((math.e ** (-lyambda * a[number])) - (math.e ** (-lyambda * a[number + 1])))
    n_P.append(n * P[number])
    m_mines_m.append(m[number] - n_P[number])
    number += 1
else:
    number = 6

buff_num1 = 0
buff_num2 = 0
buff_arr = []

while number >= 0:
    if m[number] < 5:
        buff_num1 += m_mines_m[number]
        buff_num2 += n_P[number]
    else:
        buff_num1 += m_mines_m[number]
        buff_num2 += n_P[number]
        m_mines_m2_divide_m.append((buff_num1 ** 2) / buff_num2)
        buff_arr.append(abs(buff_num1))
        buff_num1 = 0
        buff_num2 = 0
    number -= 1
else:
    number = 0
    buff_num1 = 0
    buff_num2 = 0

m_mines_m2_divide_m.reverse()
buff_arr.reverse()
m_mines_m = buff_arr
index = 0
results = []

while number < 7:
    if m[number] < 5:
        results.append([
            f"{round(a[number], 4)} - {round(a[number + 1], 4)}",
            round(P[number], 4),
            round(n_P[number], 4),
            round(m[number], 4),
            "",
            "",
        ])
    else:
        results.append([
            f"{round(a[number], 4)} - {round(a[number + 1], 4)}",
            round(P[number], 4),
            round(n_P[number], 4),
            round(m[number], 4),
            round(m_mines_m[index], 4),
            round(m_mines_m2_divide_m[index], 4),
        ])
        index += 1
    number += 1
else:
    number = 0

df = pd.DataFrame(results, columns=["Интервалы", "Pᵢ", "mᵢ'", "mᵢ", "|mᵢ-mᵢ'|", "(mᵢ-mᵢ')²/mᵢ'"])
print(df)

print(f"Суммы: P = {round(sum(P), 4)}   m = {round(sum(m), 4)}   X²_набл = {round(sum(m_mines_m2_divide_m), 4)}")

print("По таблице критических точек распределения X²",
      "\nнаходим",
      f"\nX²_крит(a, k = l - 3) = X²_крит(0.05; {len(m_mines_m2_divide_m)} - 3) = X²_крит(0.05);",
      f"{len(m_mines_m2_divide_m) - 3}) = {round(chi2.ppf(1-.05, len(m_mines_m2_divide_m) - 3), 4)}.",
      f"\nТак как X²_набл = {round(sum(m_mines_m2_divide_m), 4)} < X²_крит, то гипотеза Н₀ о нормальном",
      "\nраспределении не отвергается.",
      "\nПо критерию Колмогорова надо сравнить",
      "\nλ_опыт = n½•max|F*(xᵢ) - F(xᵢ)| с λ_крит(0.05) = 1.358.",
      f"\nn = {n} F(aᵢ) = 1 - e^(-{round(lyambda, 4)}•aᵢ), i = 0.{len(a) - 1}.")

F_mines_F = []
results = []

for i in a:
    F_mines_F.append(abs(F_a[number] - (1 - (math.e ** (-lyambda * a[number])))))
    results.append([
        round(a[number], 4),
        round(F_a[number], 4),
        round(1 - (math.e ** (-lyambda * a[number])), 4),
        round(F_mines_F[number], 4)
    ])
    number += 1
else:
    number = 0

df = pd.DataFrame(results, columns=["aᵢ", "F*(aᵢ)", "F(aᵢ)", "|F*(aᵢ)-F(aᵢ)'|"])
print(df)

print(f"max|F*(aᵢ) - F(aᵢ)| = {round(max(F_mines_F), 4)}   λ_опыт = {n}½•{round(max(F_mines_F), 4)} =",
      f"{round((n ** (1/2)) * max(F_mines_F), 4)}, λ_опыт < λ_крит =>",
      "\nгипотеза Н₀ не отвергается.\n")

print("ЗАДАНИЕ 3.")

arr = [
    [1, 3, 4, 0, 0, 0, 0, 0],
    [0, 5, 6, 5, 0, 0, 0, 0],
    [0, 0, 4, 8, 6, 0, 0, 0],
    [0, 0, 6, 15, 9, 0, 0, 0],
    [0, 0, 0, 0, 5, 6, 7, 0],
    [0, 0, 0, 0, 0, 1, 7, 2]
]
n = 100
Y = [110, 130, 150, 170, 190, 210, 230, 250]
X = [10, 13, 16, 19, 22, 25]
m_x = [8, 16, 18, 30, 18, 10]
m_y = [1, 8, 20, 28, 20, 7, 14, 2]

print(f"Значения признаков X и Y заданы \nкорреляционной таблицей объема n = {n}.")

print("Уравнение прямой регрессии Y на X имеет вид:",
      "\nyᵪ - y = rᵦ(σᵧ/σᵪ)•(x - x_)")

print("Уравнение прямой регрессии X на Y имеет вид:",
      "\nxᵧ - x = rᵦ(σᵪ/σᵧ)•(y - y_)")

index = 0
С_1 = X[(len(X) - 1) // 2]
С_2 = Y[(len(Y) - 1) // 2]

h_1 = X[1] - X[0]
h_2 = Y[1] - Y[0]

print("Находим x, σᵪ, y, σᵧ.",
      "\nДля облегчения расчетов введем так называемые условные",
      "\nварианты uᵢ u vᵢ",
      f"\nuᵢ = (xᵢ - C₁)/h₁ = (xᵢ - {round(С_1, 4)})/{h_1}   i = 0.{len(X) - 1}",
      f"\nvⱼ = (yⱼ - C₂)/h₂ = (yⱼ - {round(С_2, 4)})/{h_2}   j = 0.{len(Y) - 1}")

i = 0
x = []
u = []
u_multiply_m = []
u2_multiply_m = []
results = []

while i < len(X):
    x.append(X[i])
    u.append((x[i] - С_1) / h_1)
    u_multiply_m.append(u[i] * m_x[i])
    u2_multiply_m.append((u[i] ** 2) * m_x[i])
    results.append([
        x[i],
        round(u[i], 4),
        round(m_x[i], 4),
        round(u_multiply_m[i], 4),
        round(u2_multiply_m[i], 4)
    ])
    i += 1
else:
    i = 0

df = pd.DataFrame(results, columns=["xᵢ", "uᵢ", "mᵪ", "uᵢ•mᵪ", "uᵢ²•mᵪ"])
print(df)

print(f"Суммы: mᵪ = {round(sum(m_x), 4)}   uᵢmᵪ = {round(sum(u_multiply_m), 4)}",
      f"  uᵢ²mᵪ = {round(sum(u2_multiply_m), 4)}")

u_B = sum(u_multiply_m) / sum(m_x)
u_2 = sum(u2_multiply_m) / sum(m_x)
D_u = u_2 - (u_B ** 2)
x_B = С_1 + h_1 * u_B
D_x = (h_1 ** 2) * D_u
б_x = D_x ** (1/2)

print(f"uᵦ = {round(sum(u_multiply_m), 4)}/{round(sum(m_x), 4)} = {round(u_B, 4)}")
print(f"u² = {round(sum(u2_multiply_m), 4)}/{round(sum(m_x), 4)} = {round(u_2, 4)}")
print(f"Dᵤ = {round(u_2, 4)} - {round(u_B ** 2, 4)} = {round(D_u, 4)}")
print(f"xᵦ = {round(С_1, 4)} + {round(h_1, 4)}•{round(u_B, 4)} = {round(x_B, 4)}")
print(f"Dᵪ = {round(h_1 ** 2, 4)}•{round(D_u, 4)} = {round(D_x, 4)}")
print(f"σᵪ = {round(б_x, 4)}")

y = []
v = []
v_multiply_m = []
v2_multiply_m = []
results = []

while i < len(Y):
    y.append(Y[i])
    v.append((y[i] - С_2) / h_2)
    v_multiply_m.append(v[i] * m_y[i])
    v2_multiply_m.append((v[i] ** 2) * m_y[i])
    results.append([
        y[i],
        round(v[i], 4),
        round(m_y[i], 4),
        round(v_multiply_m[i], 4),
        round(v2_multiply_m[i], 4)
    ])
    i += 1
else:
    i = 0

df = pd.DataFrame(results, columns=["yᵢ", "vᵢ", "mᵧ", "vⱼ•mᵧ", "vⱼ²•mᵧ"])
print(df)

print(f"Суммы: mᵧ = {round(sum(m_y), 4)}   vⱼmᵧ = {round(sum(v_multiply_m), 4)}",
      f"  vⱼ²mᵧ = {round(sum(v2_multiply_m), 4)}")

v_B = sum(v_multiply_m) / sum(m_y)
v_2 = sum(v2_multiply_m) / sum(m_y)
D_v = v_2 - (v_B ** 2)
y_B = С_2 + h_2 * v_B
D_y = (h_2 ** 2) * D_v
б_y = D_y ** (1/2)

print(f"vᵦ = {round(sum(v_multiply_m), 4)}/{round(sum(m_y), 4)} = {round(v_B, 4)}")
print(f"v² = {round(sum(v2_multiply_m), 4)}/{round(sum(m_y), 4)} = {round(v_2, 4)}")
print(f"Dᵥ = {round(v_2, 4)} - {round(v_B ** 2, 4)} = {round(D_v, 4)}")
print(f"yᵦ = {round(С_2, 4)} + {round(h_1, 4)}•{round(v_B, 4)} = {round(y_B, 4)}")
print(f"Dᵧ = {round(h_2 ** 2, 4)}•{round(D_v, 4)} = {round(D_y, 4)}")
print(f"σᵧ = {round(б_y, 4)}")

print("Находим выборочный коэффициент корреляции",
      "\nuv = (∑∑uᵢvⱼmᵪᵧ)/n"
      "\nrᵦ = rᵪᵧ = (xy - x•y)/σᵪ•σᵧ = rᵤᵥ = (uv - u•v)/σᵤ•σᵥ")

sum_i = []
sum_j = []

for index in X:
    sum_i.append(0)

for index in Y:
    sum_j.append(0)

j = 0
i = 0
v_u_m = []
u_v_m = []

while i < len(X):
    u_v_m.append(0)
    while j < len(Y):
        u_v_m[i] += arr[i][j] * v[j]
        j += 1
    else:
        j = 0
    u_v_m[i] = u_v_m[i] * u[i]
    i += 1
else:
    i = 0

while i < len(Y):
    v_u_m.append(0)
    while j < len(X):
        v_u_m[i] += arr[j][i] * u[j]
        j += 1
    else:
        j = 0
    v_u_m[i] = v_u_m[i] * v[i]
    i += 1
else:
    i = 0

results = []

while i < len(Y):
    if i < len(X):
        results.append([
            round(v_u_m[i], 4),
            round(u_v_m[i], 4)
        ])
    else:
        results.append([
            round(v_u_m[i], 4),
            ""
        ])
    i += 1
else:
    i = 0

df = pd.DataFrame(results, columns=["vⱼ∑uᵢmᵪᵧ", "uᵢ∑vⱼmᵪᵧ"])
print(df)

uv = sum(v_u_m) / n
uv_mines_uv = uv - u_B * v_B
б_u = D_u ** (1/2)
б_v = D_v ** (1/2)
r_B = uv_mines_uv / (б_u * б_v)

print(f"uv = {round(sum(v_u_m), 4)}/{round(n, 4)} = {round(uv, 4)}")
print(f"uv - u•v = {round(uv, 4)} - {round(u_B, 4)}•{round(v_B, 4)} = {round(uv_mines_uv, 4)}")
print(f"σᵤ = {round(б_u, 4)}")
print(f"σᵥ = {round(б_v, 4)}")
print(f"rᵦ = {round(uv_mines_uv, 4)}/({round(h_1, 4)}•{round(б_v, 4)}) = {round(r_B, 4)}")

print(f"Близость |rᵦ| = {abs(round(r_B, 4))} к 1 говорит о достаточно тесной линейной",
      "\nзависимости между СВ Х и Y; т.к. с возрастанием значений одной",
      "\nслучайной величины значения другой СВ убывают, то rᵦ < 0.",
      "\nОтметим, что вычисления, записанные в трех таблицах, можно",
      "\nсвести в одну таблицу.",
      "\nВыпишем уравнения прямых регрессии (1) и (2):")

print(f"Y на X:   yᵪ = {round(r_B, 4)}•({round(б_y, 4)}/{round(б_x, 4)})•(x - {round(x_B, 4)}) + {round(y_B, 4)}  (3)")

print(f"X на Y:   xᵧ = {round(r_B, 4)}•({round(б_x, 4)}/{round(б_y, 4)})•(y - {round(y_B, 4)}) + {round(x_B, 4)}  (4)")

print("На плоскости хОy строим графики прямых (3) и (4) и значения \n(Х,Y) из корреляционной таблицы.",
      "\n(Вывод графика)")

grafik_x = []
grafik_y = []

number = 0
results = []
results1 = []

for i in X:
    grafik_y.append(r_B * (б_y / б_x) * (i - x_B) + y_B)
    results.append(
        f"x = {round(i, 4)}   yᵪ = {round(grafik_y[number], 4)}"
    )
    number += 1
else:
    number = 0

for i in Y:
    grafik_x.append(r_B * (б_x / б_y) * (i - y_B) + x_B)
    results1.append(
        f"y = {round(i, 4)}   xᵧ = {round(grafik_x[number], 4)}"
    )
    number += 1
else:
    number = 0


max_length = max(len(results), len(results1))
results += [""] * (max_length - len(results))
results1 += [""] * (max_length - len(results1))
df = pd.DataFrame({" ": results1, "  ": results})
print(df.to_string(index=False))

plt.clf()
plt.close("all")

plt.plot(X, grafik_y, marker='.')
plt.plot(grafik_x, Y, marker='.')

points_X = []
points_Y = []

i = 0
j = 0

while i < len(arr):
    while j < len(arr[0]):
        if arr[i][j] != 0:
            points_X.append(X[i])
            points_Y.append(Y[j])
        j += 1
    else:
        j = 0
    i += 1
else:
    i = 0

plt.scatter(points_X, points_Y, color='black', s=10)

plt.xlabel("X")
plt.ylabel("Y")

plt.show()

print("Оценим значимость выборочного коэффициента корреляции",
      f"\nrᵦ = {round(r_B, 4)} для генеральной совокупности (X, Y) при заданном уровне",
      "значимости a = 0.05.",
      "\nВыдвигаем нулевую и альтернативную гипотезы:",
      "\nН₀: r_Г = 0 (в генеральной совокупности нет линейной зависимости).",
      "\nН₁: r_Г ≠ 0 (в генеральной совокупности есть линейная зависимость между СВ Х и Y).",
      "\nНаходим значение выборочной статистики")

print(f"|t_набл| = |rᵦ|•(n-2)½ / (1 - rᵦ²)½ = {round(abs(r_B), 4)}•({n - 2})½ / (1 - {round(abs(r_B), 4)}²)½ =",
      round((abs(r_B) * ((n - 2) ** (1/2))) / ((1 - (r_B ** 2)) ** (1/2)), 4))

print("По таблице находим:",
      f"\nt_крит(a; n - 2) = t_крит(0.05; {n - 2}) = {round(t.ppf(1-0.05, n - 2), 4)}")

print(f"|t_набл| > {round(t.ppf(1-0.05, n - 2), 4)} Н₀ отвергаем и принимаем гипотезу Н₁.",
      f"\nСледовательно, rᵦ = {round(r_B, 4)} - значимый коэффициент.")
