import math
from scipy.stats import norm
import matplotlib.pyplot as plt

print("ЗАДАНИЕ 1.", "\nВ результате эксперимента получена выборка из 100 чисел: ")

experimental_values = [
    24.8, 26.2, 25.6, 24.0, 26.4, 25.2, 26.7, 25.4, 25.3, 26.1,
    24.3, 25.3, 25.6, 26.7, 24.5, 26.0, 25.7, 25.0, 26.4, 25.9,
    24.4, 25.4, 26.1, 23.4, 26.5, 25.9, 23.9, 25.7, 27.1, 24.9,
    23.8, 25.6, 25.2, 26.4, 24.2, 26.5, 25.7, 24.7, 26.0, 25.8,
    24.3, 25.5, 26.7, 24.9, 26.2, 26.7, 24.6, 26.0, 25.4, 25.0,
    25.4, 25.3, 24.1, 26.6, 24.8, 25.6, 23.7, 26.8, 25.2, 26.1,
    24.5, 25.4, 25.1, 26.2, 24.2, 26.4, 25.7, 23.9, 27.2, 25.0,
    23.9, 25.6, 24.9, 24.5, 26.2, 26.7, 24.3, 26.1, 27.7, 25.8,
    25.6, 25.2, 24.2, 26.0, 24.7, 26.5, 23.5, 25.4, 27.1, 24.0,
    26.2, 24.2, 25.5, 26.0, 25.7, 26.4, 24.6, 27.0, 25.2, 26.9
]

number = 1

for num in experimental_values:
    print(num, " ", end="")
    if number % 10 == 0:
        print()
    number += 1
else:
    number = 1
    print()

experimental_values.sort()

print("1. Записываем числовые значения (варианты) в порядке возрастания, получим вариационный ряд: ")

for num in experimental_values:
    print(num, " ", end="")
    if number % 10 == 0:
        print()
    number += 1
else:
    number = 0
    print()

scope_of_variation = experimental_values[99] - experimental_values[0]

print("2. Находим размах вариации: \nxₘₐₓ - xₘᵢₙ = ", experimental_values[99], " - ", experimental_values[0], " = ",
      round(scope_of_variation, 4), end="\n")

optimal_number_of_intervals_k = round(1 + 3.322 * math.log10(100))

print("оптимальное число интервалов: \nk = 1 + 3.322lg(n) = 1 + 3.322lg(100) = ", 1 + 3.322 * math.log10(100), " ≈ ",
      optimal_number_of_intervals_k, end="\n")

partial_interval_length_h = scope_of_variation / 8

print("и длину частичного интервала: \nh = (xₘₐₓ - xₘᵢₙ)/8 = ", round(scope_of_variation, 4), "/", 8, " = ",
      round(partial_interval_length_h, 4), end="\n")

interval_boundaries = [experimental_values[0]]

while number < 8:
    interval_boundaries.append(interval_boundaries[number] + partial_interval_length_h)
    number += 1
else:
    number = 0

print("Выпишем границы интервалов",
      "\na₁ = ", round(interval_boundaries[0], 4), "; a₂ = ", round(interval_boundaries[1], 4), "; a₃ = ",
      round(interval_boundaries[2], 4), "; a₄ = ", round(interval_boundaries[3], 4), "; a₅ = ",
      round(interval_boundaries[4], 4), "; \na₆ = ", round(interval_boundaries[5], 4), "; a₇ = ",
      round(interval_boundaries[6], 4), "; a₈ = ", round(interval_boundaries[7], 4), "; a₉ = ",
      round(interval_boundaries[8], 4), ";")

print("Подсчитаем число вариант, попавших в каждый интервал, т.е. находим частоты mᵢ , запишем интервальное,",
      "\nраспределение частот выборки:")

distribution_of_sampling_frequencies_m = [0]

for num in experimental_values:
    if (num >= interval_boundaries[number]) and (num <= interval_boundaries[number + 1]):
        distribution_of_sampling_frequencies_m[number] += 1
    else:
        distribution_of_sampling_frequencies_m.append(1)
        print(round(interval_boundaries[number], 4), "-", round(interval_boundaries[number + 1], 4),
              f" m{number + 1} = ", round(distribution_of_sampling_frequencies_m[number], 4))
        number += 1
else:
    print(round(interval_boundaries[number], 4), "-", round(interval_boundaries[number + 1], 4),
          f" m{number + 1} = ", round(distribution_of_sampling_frequencies_m[number], 4), "\n")
    number = 0

number_of_experimental_values_n = len(experimental_values)

print("3. Находим относительные частоты: \nwᵢ = mᵢ/n, n = ", number_of_experimental_values_n,
      "\nи их плотности \nwᵢ/h, h = ", round(partial_interval_length_h, 4))

relative_frequencies_w = []
densities = []


for num in distribution_of_sampling_frequencies_m:
    relative_frequencies_w.append(num / number_of_experimental_values_n)
    densities.append(relative_frequencies_w[number] / partial_interval_length_h)
    print(f"{round(interval_boundaries[number], 4)} - {round(interval_boundaries[number + 1], 4)} ",
          f"m{number + 1} = {round(num, 4)} w{number + 1} = {round(relative_frequencies_w[number], 4)} ",
          f"w{number + 1}/h = {round(densities[number], 4)};")
    number += 1
else:
    number = 0

print("Строим гистограмму относительных частот (масштаб на осях разный).\n(график на экране)")

fig, ax = plt.subplots()
ax.hist(interval_boundaries[:-1], weights=densities, bins=interval_boundaries)

ax.set_xlim(min(interval_boundaries), max(interval_boundaries))
ax.set_ylim(min(densities), max(densities))

ax.set_title("Гистограмма относительных частот")
ax.set_xlabel("Интервалы")
ax.set_ylabel("wᵢ/h")

#plt.show()

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
plt.plot(interval_boundaries, values_of_the_empirical_function_F, marker='o', mec='r', mfc='w')
plt.xlim(min(interval_boundaries), max(interval_boundaries))
plt.ylim(min(values_of_the_empirical_function_F), max(values_of_the_empirical_function_F))

plt.title("График эмпирической функции")
plt.xlabel("Интервалы")
plt.ylabel("F(x)")

#plt.show()

print("4. Находим выборочное среднее, среднее по квадратам, выборочную дисперсию,\nсреднее квадратическое отклонение, ",
      "исправленную выборочную дисперсию,\nисправленное среднее квадратическое отклонение.",
      "\nСоставим расчетную таблицу.")

xi_mi = 0
xi2_mi = 0

while number < len(interval_boundaries) - 1:
    xi_mi += ((interval_boundaries[number] + interval_boundaries[number + 1]) / 2) \
             * distribution_of_sampling_frequencies_m[number]
    xi2_mi += (((interval_boundaries[number] + interval_boundaries[number + 1]) / 2)
               ** 2) * distribution_of_sampling_frequencies_m[number]
    print(f"{round(interval_boundaries[number], 4)}-{round(interval_boundaries[number + 1], 4)} ",
          f"x{number + 1} = {round((interval_boundaries[number] + interval_boundaries[number + 1]) / 2, 4)} ",
          f"m{number + 1} = {round(distribution_of_sampling_frequencies_m[number], 4)}", f"x{number + 1}•m{number + 1} = ",
          round(((interval_boundaries[number] + interval_boundaries[number + 1]) / 2)
          * distribution_of_sampling_frequencies_m[number], 4), " ", f"(x{number + 1}^2)•m{number + 1} = ",
          round((((interval_boundaries[number] + interval_boundaries[number + 1]) / 2)
           ** 2) * distribution_of_sampling_frequencies_m[number], 4))
    number += 1
else:
    number = 0

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
print(f"s² = (100/99)*{round(sample_variance_D, 4)} = {round(corrected_sample_variance_s2, 4)}")
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
P = []

for num in interval_boundaries:
    if num == interval_boundaries[-1]:
        print(f"a{number} = {round(num, 4)} (a{number} - x)/s =",
              f"{round((num - sample_mean_x) / corrected_mean_square_deviation_s, 4)}",
              f" Ф(u{number}) = {round(norm.cdf((num - sample_mean_x) / corrected_mean_square_deviation_s) - 0.5, 4)}",
              f" P(сумма) = {round(buff_num, 4)}")
    else:
        print(f"a{number} = {round(num, 4)} (a{number} - x)/s =",
              f"{round((num - sample_mean_x) / corrected_mean_square_deviation_s, 4)}",
              f" Ф(u{number}) = {round(norm.cdf((num - sample_mean_x) / corrected_mean_square_deviation_s) - 0.5, 4)}",
              f" P{number} =",
              round((norm.cdf((interval_boundaries[number] - sample_mean_x) / corrected_mean_square_deviation_s) - 0.5)
                    - (norm.cdf((num - sample_mean_x) / corrected_mean_square_deviation_s) - 0.5),4))
        buff_num += (norm.cdf((interval_boundaries[number] - sample_mean_x) / corrected_mean_square_deviation_s)
                     - 0.5) - (norm.cdf((num - sample_mean_x) / corrected_mean_square_deviation_s) - 0.5)
        P.append((norm.cdf((interval_boundaries[number] - sample_mean_x) / corrected_mean_square_deviation_s)
                  - 0.5) - (norm.cdf((num - sample_mean_x) / corrected_mean_square_deviation_s) - 0.5))
    number += 1
else:
    number = 0

print("Статистика имеет распределение «хи-квадрат» лишь при n → ∞, поэтому необходимо, чтобы в каждом",
      "\nинтервале было не менее 5 значений. Если mᵢ < 5, имеет смысл объединить соседние интервалы.",
      "\nВ данном случае объединять не будем, так как все вычисления производятся программно l = 8.")

derivative_of_m = []
buff_num = 0
x2_nabl = 0

for num in distribution_of_sampling_frequencies_m:
    derivative_of_m.append(100 * P[number])
    buff_num = num - derivative_of_m[number]
    x2_nabl += (buff_num ** 2) / derivative_of_m[number]
    print(f"№{number + 1} m{number + 1} = {num} P{number + 1} = m{number + 1}' = {round(derivative_of_m[number], 4)}",
          f"m{number + 1} - m{number + 1}' = {round(buff_num, 4)} (m{number + 1} - m{number + 1}')² =",
          f"{round(buff_num ** 2, 4)}", f"((m{number + 1} - m{number + 1}')²)/m{number + 1}' =",
          round((buff_num ** 2) / derivative_of_m[number], 4))
    number += 1
else:
    print(f"m(сумма)' = {round(sum(derivative_of_m), 4)} x²_набл = {round(x2_nabl, 4)}")
    number = 0

