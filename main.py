import math
import matplotlib.pyplot as plt

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
      scope_of_variation, end="\n")

optimal_number_of_intervals_k = round(1 + 3.322 * math.log10(100))

print("оптимальное число интервалов: \nk = 1 + 3.322lg(n) = 1 + 3.322lg(100) = ", 1 + 3.322 * math.log10(100), " ≈ ",
      optimal_number_of_intervals_k, end="\n")

partial_interval_length_h = scope_of_variation / 8

print("и длину частичного интервала: \nh = (xₘₐₓ - xₘᵢₙ)/8 = ", scope_of_variation, "/", 8, " = ",
      partial_interval_length_h, end="\n")

interval_boundaries = [experimental_values[0]]

while number < 8:
    interval_boundaries.append(interval_boundaries[number] + partial_interval_length_h)
    number += 1
else:
    number = 0

print("Выпишем границы интервалов",
      "\na₁ = ", interval_boundaries[0], "; a₂ = ", interval_boundaries[1], "; a₃ = ", interval_boundaries[2],
      "; a₄ = ", interval_boundaries[3], "; a₅ = ", interval_boundaries[4], "; \na₆ = ", interval_boundaries[5],
      "; a₇ = ", interval_boundaries[6], "; a₈ = ", interval_boundaries[7], "; a₉ = ", interval_boundaries[8], ";")

print("Подсчитаем число вариант, попавших в каждый интервал, т.е. находим частоты mᵢ , запишем интервальное,",
      "\nраспределение частот выборки:")

distribution_of_sampling_frequencies_m = [0]

for num in experimental_values:
    if (num >= interval_boundaries[number]) and (num <= interval_boundaries[number + 1]):
        distribution_of_sampling_frequencies_m[number] += 1
    else:
        distribution_of_sampling_frequencies_m.append(1)
        number += 1
else:
    number = 0

print(interval_boundaries[0], "-", interval_boundaries[1], " m₁ = ", distribution_of_sampling_frequencies_m[0], ";  ",
      interval_boundaries[1], "-", interval_boundaries[2], " m₂ = ", distribution_of_sampling_frequencies_m[1], ";  ",
      interval_boundaries[2], "-", interval_boundaries[3], " m₃ = ", distribution_of_sampling_frequencies_m[2], ";  ",
      interval_boundaries[3], "-", interval_boundaries[4], " m₄ = ", distribution_of_sampling_frequencies_m[3], ";\n",
      interval_boundaries[4], "-", interval_boundaries[5], " m₅ = ", distribution_of_sampling_frequencies_m[4], ";  ",
      interval_boundaries[5], "-", interval_boundaries[6], " m₆ = ", distribution_of_sampling_frequencies_m[5], ";  ",
      interval_boundaries[6], "-", interval_boundaries[7], " m₇ = ", distribution_of_sampling_frequencies_m[6], ";  ",
      interval_boundaries[7], "-", interval_boundaries[8], " m₈ = ", distribution_of_sampling_frequencies_m[7], ";\n")

number_of_experimental_values_n = len(experimental_values)

print("3. Находим относительные частоты: \nwᵢ = mᵢ/n, n = ", number_of_experimental_values_n,
      "\nи их плотности wᵢ/h, h = ", partial_interval_length_h)

relative_frequencies_w = []
densities = []


for num in distribution_of_sampling_frequencies_m:
    relative_frequencies_w.append(num / number_of_experimental_values_n)
    densities.append(relative_frequencies_w[number] / partial_interval_length_h)
    number += 1
else:
    number = 0

print(interval_boundaries[0], "-", interval_boundaries[1], " m₁ = ", distribution_of_sampling_frequencies_m[0],
      " w₁ = ", relative_frequencies_w[0], " w₁/h = ", densities[0], ";\n",
      interval_boundaries[1], "-", interval_boundaries[2], " m₂ = ", distribution_of_sampling_frequencies_m[1],
      " w₂ = ", relative_frequencies_w[1], " w₂/h = ", densities[1], ";\n",
      interval_boundaries[2], "-", interval_boundaries[3], " m₃ = ", distribution_of_sampling_frequencies_m[2],
      " w₃ = ", relative_frequencies_w[2], " w₃/h = ", densities[2], ";\n",
      interval_boundaries[3], "-", interval_boundaries[4], " m₄ = ", distribution_of_sampling_frequencies_m[3],
      " w₄ = ", relative_frequencies_w[3], " w₄/h = ", densities[3], ";\n",
      interval_boundaries[4], "-", interval_boundaries[5], " m₅ = ", distribution_of_sampling_frequencies_m[4],
      " w₅ = ", relative_frequencies_w[4], " w₅/h = ", densities[4], ";\n",
      interval_boundaries[5], "-", interval_boundaries[6], " m₆ = ", distribution_of_sampling_frequencies_m[5],
      " w₆ = ", relative_frequencies_w[5], " w₆/h = ", densities[5], ";\n",
      interval_boundaries[6], "-", interval_boundaries[7], " m₇ = ", distribution_of_sampling_frequencies_m[6],
      " w₇ = ", relative_frequencies_w[6], " w₇/h = ", densities[6], ";\n",
      interval_boundaries[7], "-", interval_boundaries[8], " m₈ = ", distribution_of_sampling_frequencies_m[7],
      " w₈ = ", relative_frequencies_w[7], " w₈/h = ", densities[7], ";")

print("Строим гистограмму относительных частот (масштаб на осях разный).\n(график на экране)")

fig, ax = plt.subplots()
ax.hist(interval_boundaries[:-1], weights=densities, bins=interval_boundaries)

ax.set_xlim(0, max(interval_boundaries)+15)
ax.set_ylim(0, max(densities) + 0.001)

ax.set_title("Гистограмма относительных частот")
ax.set_xlabel("Интервалы")
ax.set_ylabel("wᵢ/h")

#plt.show()

print("Находим значения эмпирической функции:", "\nF(x)=nₓ/n, где nₓ - накопленная частота.",
      "\nВ качестве аргумента функции рассматриваем концы интервалов:")

values_of_the_empirical_function_F = [0]
print(f"F({interval_boundaries[0]}) = {values_of_the_empirical_function_F[0]}/{100} = {0}")

buff_num = 0
for num in distribution_of_sampling_frequencies_m:
    values_of_the_empirical_function_F.append((buff_num + num) / 100)
    print(f"F({interval_boundaries[number + 1]}) = ({buff_num} + {num})/{100} =",
          f"{values_of_the_empirical_function_F[number + 1]}")
    buff_num += num
    number += 1
else:
    number = 0
    buff_num = 0

print("Строим график эмпирической функции или кумулятивной кривой выборки:\n(график на экране)\n")

plt.clf()
plt.plot(interval_boundaries, values_of_the_empirical_function_F, marker='o', mec='r', mfc='w')
plt.xlim(0, max(interval_boundaries)+15)
plt.ylim(0, max(values_of_the_empirical_function_F) + 0.1)

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
    print(f"{interval_boundaries[number]}-{interval_boundaries[number + 1]} ",
          f"x{number + 1} = {((interval_boundaries[number] + interval_boundaries[number + 1]) / 2)} ",
          f"m{number + 1} = {distribution_of_sampling_frequencies_m[number]}",
          f"x{number + 1}*m{number + 1} = ", ((interval_boundaries[number] + interval_boundaries[number + 1]) / 2)
          * distribution_of_sampling_frequencies_m[number], " ",
          f"(x{number + 1}^2)*m{number + 1} = ",
          (((interval_boundaries[number] + interval_boundaries[number + 1]) / 2)
           ** 2) * distribution_of_sampling_frequencies_m[number])
    number += 1
else:
    number = 0

print(f"Сумма m = 100 x*m = {xi_mi} x^2*m = {xi2_mi}")

sample_mean_x = xi_mi / 100
squared_mean_x2 = xi2_mi / 100
sample_variance_D = squared_mean_x2 - (sample_mean_x ** 2)
mean_square_deviation_G = math.sqrt(sample_variance_D)
corrected_sample_variance_s2 = (100 / 99) * sample_variance_D
corrected_mean_square_deviation_s = math.sqrt(corrected_sample_variance_s2)

print(f"xᵦ = {xi_mi}/100 = {sample_mean_x}")
print(f"x² = {xi2_mi}/100 = {squared_mean_x2}")
print(f"Dᵦ = {squared_mean_x2}-{sample_mean_x}^2 = {sample_variance_D}")
print(f"σᵦ = {sample_variance_D}^(1/2) = {mean_square_deviation_G}")
print(f"s² = (100/99)*{sample_variance_D} = {corrected_sample_variance_s2}")
print(f"s = {corrected_sample_variance_s2}^(1/2) = {corrected_mean_square_deviation_s}\n")

print("5. По виду гистограммы выдвигаем гипотезу о нормальном распределении",
      "\nв генеральной совокупности признака Х, причем")
