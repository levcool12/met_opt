import math


# ============================================================
# Функция
# ============================================================

def f(x):
    return x**2 + math.exp(x)


# ============================================================
# 1. Метод полного перебора
# ============================================================

def brute_force(a, b, eps):
    n = int(round((b - a) / eps)) + 1

    best_x = a
    best_f = f(a)
    N = 1

    for i in range(1, n):
        x = a + i * eps
        fx = f(x)
        N += 1

        if fx < best_f:
            best_x = x
            best_f = fx

    return best_x, best_f, N


# ============================================================
# 2. Метод парабол
# ============================================================

def parabolic_method(a, b, eps):
    # Начальные три точки
    x1 = a
    x2 = (a + b) / 2
    x3 = b

    f1 = f(x1)
    f2 = f(x2)
    f3 = f(x3)

    N = 3

    while True:
        # Координата вершины параболы,
        # проходящей через три точки
        numerator = (
            (x2 - x1) ** 2 * (f2 - f3)
            - (x2 - x3) ** 2 * (f2 - f1)
        )

        denominator = (
            2 * (
                (x2 - x1) * (f2 - f3)
                - (x2 - x3) * (f2 - f1)
            )
        )

        if abs(denominator) < 1e-15:
            break

        x_new = x2 - numerator / denominator

        # Если новая точка вышла за интервал,
        # используем середину
        if not (min(x1, x3) < x_new < max(x1, x3)):
            x_new = (x1 + x3) / 2

        f_new = f(x_new)
        N += 1

        # Проверка точности
        if abs(x_new - x2) < eps:
            return x_new, f_new, N

        # Собираем три точки вокруг минимума
        points = [
            (x1, f1),
            (x2, f2),
            (x3, f3),
            (x_new, f_new)
        ]

        points.sort(key=lambda p: p[0])

        # Ищем точку с минимальным значением
        min_index = min(range(4), key=lambda i: points[i][1])

        # Если минимум оказался внутри,
        # берем ближайшие точки слева и справа
        if 0 < min_index < 3:
            left = points[min_index - 1]
            center = points[min_index]
            right = points[min_index + 1]

            x1, f1 = left
            x2, f2 = center
            x3, f3 = right

        else:
            # Запасной вариант
            points.sort(key=lambda p: p[1])
            x2, f2 = points[0]

            remaining = sorted(
                points[1:],
                key=lambda p: abs(p[0] - x2)
            )

            x1, f1 = remaining[0]
            x3, f3 = remaining[1]

        if abs(x3 - x1) <= eps:
            return x2, f2, N

    return x2, f2, N


# ============================================================
# 3. Метод золотого сечения
# ============================================================

def golden_section(a, b, eps):
    phi = (math.sqrt(5) - 1) / 2

    x1 = b - phi * (b - a)
    x2 = a + phi * (b - a)

    f1 = f(x1)
    f2 = f(x2)

    N = 2

    while abs(b - a) > eps:
        if f1 <= f2:
            b = x2
            x2 = x1
            f2 = f1

            x1 = b - phi * (b - a)
            f1 = f(x1)

            N += 1

        else:
            a = x1
            x1 = x2
            f1 = f2

            x2 = a + phi * (b - a)
            f2 = f(x2)

            N += 1

    x_best = (a + b) / 2

    return x_best, f(x_best), N + 1


# ============================================================
# 4. Метод дихотомии I порядка
# ============================================================

def dichotomy(a, b, eps):
    # Малое смещение относительно середины
    delta = eps / 3

    N = 0

    while (b - a) > eps:
        x_mid = (a + b) / 2

        x1 = x_mid - delta
        x2 = x_mid + delta

        f1 = f(x1)
        f2 = f(x2)

        N += 2

        if f1 < f2:
            b = x2
        else:
            a = x1

    x_best = (a + b) / 2

    return x_best, f(x_best), N + 1


# ============================================================
# Вывод результатов
# ============================================================

def print_results(eps):
    a = -1
    b = 0

    methods = [
        ("Полный перебор", brute_force),
        ("Метод парабол", parabolic_method),
        ("Золотое сечение", golden_section),
        ("Дихотомия I порядка", dichotomy)
    ]

    print()
    print("=" * 80)
    print(f"epsilon = {eps}")
    print("=" * 80)

    print(
        f"{'Метод':<25}"
        f"{'x*':>18}"
        f"{'f(x*)':>18}"
        f"{'N':>10}"
    )

    print("-" * 80)

    for name, method in methods:
        x, fx, N = method(a, b, eps)

        print(
            f"{name:<25}"
            f"{x:>18.10f}"
            f"{fx:>18.10f}"
            f"{N:>10}"
        )


# ============================================================
# Главная программа
# ============================================================

if __name__ == "__main__":

    print("Минимизация f(x) = x^2 + e^x")
    print("Интервал: [-1, 0]")

    print_results(10**-3)
    print_results(10**-4)

    print()
    print("=" * 80)
    print("Для проверки:")
    print(f"x точного минимума ≈ {-0.3517337112:.10f}")
    print(f"f(x) ≈ {f(-0.3517337112):.10f}")
    print("=" * 80)