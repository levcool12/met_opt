
import math
import os
import matplotlib

# Сохраняем графики без открытия окон
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# ИСХОДНЫЕ ДАННЫЕ
# ============================================================

A = -10.0
B = 0.0

TOLERANCES = [1e-3, 1e-4]
L = 2.0

RESULT_FILE = "result.txt"
GRAPH_DIR = "results"

# Начальные приближения для метода Ньютона
NEWTON_STARTS = [-9.0, -7.0, -5.0, -3.0, -1.0]


# ============================================================
# ФУНКЦИЯ И ПРОИЗВОДНЫЕ
# ============================================================

def f(x):
    return x + x**2 / 10 + math.sin(x)


def df(x):
    return 1 + x / 5 + math.cos(x)


def d2f(x):
    return 0.2 - math.sin(x)


# ============================================================
# МЕТОД ЛОМАНЫХ
# Метод Пиявского — Шуберта
# ============================================================

def broken_lines(a, b, eps, L=2.0, max_iter=100000):
    xs = [a, b]
    ys = [f(a), f(b)]
    history = []

    for iteration in range(max_iter):
        points = sorted(zip(xs, ys))
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        best_i = min(range(len(ys)), key=lambda i: ys[i])
        best_x, best_y = xs[best_i], ys[best_i]

        candidates = []

        for i in range(len(xs) - 1):
            x1, x2 = xs[i], xs[i + 1]
            y1, y2 = ys[i], ys[i + 1]

            # Минимум нижней оценки на данном интервале
            x_new = (x1 + x2) / 2 - (y2 - y1) / (2 * L)
            lower_y = (y1 + y2) / 2 - L * (x2 - x1) / 2

            candidates.append((lower_y, x_new, i))

        lower_y, x_new, interval_i = min(
            candidates, key=lambda item: item[0]
        )

        history.append({
            "x": x_new,
            "y": f(x_new),
            "best_x": best_x,
            "best_y": best_y,
            "lower_y": lower_y
        })

        # Остановка по разнице лучшего значения и нижней оценки
        if best_y - lower_y <= eps:
            return best_x, best_y, history, xs, ys, len(xs)

        left = xs[interval_i]
        right = xs[interval_i + 1]

        # Защита от выхода за границы интервала
        x_new = max(left, min(x_new, right))

        # Защита от повторного вычисления той же точки
        if any(abs(x_new - old_x) < 1e-14 for old_x in xs):
            x_new = (left + right) / 2

        if any(abs(x_new - old_x) < 1e-14 for old_x in xs):
            break

        xs.append(x_new)
        ys.append(f(x_new))

    best_i = min(range(len(ys)), key=lambda i: ys[i])
    return xs[best_i], ys[best_i], history, xs, ys, len(xs)


# ============================================================
# МЕТОД НЬЮТОНА
# ============================================================

def newton(x0, eps, max_iter=100):
    x = x0
    history = [x]
    status = "Достигнут лимит итераций"

    for _ in range(max_iter):
        first = df(x)
        second = d2f(x)

        if abs(second) < 1e-10:
            status = "Остановка: f''(x) близка к нулю"
            break

        x_new = x - first / second

        if not math.isfinite(x_new) or abs(x_new) > 1e6:
            status = "Расходимость: слишком большой шаг"
            break

        history.append(x_new)

        if abs(x_new - x) < eps:
            x = x_new
            status = "Сошёлся по изменению x"
            break

        x = x_new

    if A <= x <= B and abs(df(x)) < 1e-5:
        if d2f(x) > 0:
            status += "; локальный минимум"
        elif d2f(x) < 0:
            status += "; локальный максимум"
        else:
            status += "; классификация неоднозначна"
    elif not (A <= x <= B):
        status += "; результат вне отрезка"

    y = f(x) if math.isfinite(x) else float("nan")
    return x, y, history, status


# ============================================================
# ГРАФИК МЕТОДА ЛОМАНЫХ
# ============================================================

def plot_broken_lines(best_x, best_y, history, xs, ys, eps):
    fig, ax = plt.subplots(figsize=(12, 7))

    grid = [A + i * (B - A) / 2000 for i in range(2001)]
    ax.plot(grid, [f(x) for x in grid],
            label="Исходная функция f(x)", linewidth=2)

    points = sorted(zip(xs, ys))

    # Рисуем нижние ломаные на каждом исследованном интервале
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]

        xc = (x1 + x2) / 2 - (y2 - y1) / (2 * L)
        xc = max(x1, min(xc, x2))
        yc = (y1 + y2) / 2 - L * (x2 - x1) / 2

        ax.plot([x1, xc, x2], [y1, yc, y2],
                "--", color="gray", alpha=0.3)

    ax.scatter(xs, ys, s=20, label="Вычисленные точки")

    if history:
        hx = [item["x"] for item in history]
        hy = [item["y"] for item in history]

        ax.scatter(
            hx, hy,
            c=list(range(len(hx))),
            cmap="viridis",
            s=35,
            label="Новые точки (цвет — порядок)"
        )

        for i, (x, y) in enumerate(zip(hx, hy)):
            if i < 25 or i == len(hx) - 1:
                ax.annotate(
                    str(i + 1), (x, y),
                    xytext=(3, 5),
                    textcoords="offset points",
                    fontsize=7
                )

    ax.scatter(
        [best_x], [best_y],
        color="red", marker="*", s=200,
        label="Лучший найденный минимум"
    )

    ax.set_title(f"Метод ломаных, epsilon = {eps}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    filename = f"broken_lines_eps_{eps:.0e}.png"
    fig.savefig(os.path.join(GRAPH_DIR, filename), dpi=150)
    plt.close(fig)


# ============================================================
# ГРАФИК МЕТОДА НЬЮТОНА
# ============================================================

def plot_newton(results, eps):
    fig, ax = plt.subplots(figsize=(12, 7))

    grid = [A + i * (B - A) / 2000 for i in range(2001)]
    ax.plot(grid, [f(x) for x in grid],
            label="Исходная функция f(x)", linewidth=2)

    for x0, result in results.items():
        x_final, y_final, history, status = result

        # Рисуем только точки внутри исходного интервала.
        # Не соединяем через линию точки, если между ними
        # траектория вышла за границы отрезка.
        segment = []

        def draw_segment(points, label):
            if not points:
                return
            ax.plot(
                [p[0] for p in points],
                [p[1] for p in points],
                marker="o",
                markersize=4,
                label=label
            )
            for i, (x, y) in enumerate(points):
                ax.annotate(
                    str(i), (x, y),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8
                )

        for i, x in enumerate(history):
            if A <= x <= B:
                segment.append((x, f(x)))
            else:
                draw_segment(
                    segment,
                    f"x0={x0:g}" if not segment else "_nolegend_"
                )
                segment = []

        draw_segment(
            segment,
            f"x0={x0:g}" if segment else "_nolegend_"
        )

        if A <= x_final <= B and math.isfinite(y_final):
            ax.scatter([x_final], [y_final], marker="x", s=100)

    ax.set_title(f"Метод Ньютона, epsilon = {eps}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    filename = f"newton_eps_{eps:.0e}.png"
    fig.savefig(os.path.join(GRAPH_DIR, filename), dpi=150)
    plt.close(fig)


# ============================================================
# ЗАПИСЬ РЕЗУЛЬТАТОВ В result.txt
# ============================================================

def write_results():
    with open(RESULT_FILE, "w", encoding="utf-8") as file:
        file.write("Минимизация f(x) = x + x^2/10 + sin(x)\n")
        file.write("Интервал: [-10, 0]\n")

        for eps in TOLERANCES:
            file.write("\n" + "=" * 85 + "\n")
            file.write(f"epsilon = {eps}\n")
            file.write("=" * 85 + "\n")

            file.write("\nМетод ломаных\n")
            file.write("-" * 85 + "\n")
            file.write(
                f"{'x*':>18} {'f(x*)':>18} "
                f"{'N вычисленных точек':>25}\n"
            )

            bx, by, bhistory, bxs, bys, N = broken_lines(
                A, B, eps, L=L
            )

            file.write(f"{bx:18.10f} {by:18.10f} {N:25d}\n")

            file.write("\nМетод Ньютона\n")
            file.write("-" * 100 + "\n")
            file.write(
                f"{'x0':>8} {'x результата':>18} "
                f"{'f(x)':>18} {'N итераций':>12}  Статус\n"
            )

            results = {}

            for x0 in NEWTON_STARTS:
                result = newton(x0, eps)
                results[x0] = result

                x, y, history, status = result
                file.write(
                    f"{x0:8.2f} {x:18.10f} "
                    f"{y:18.10f} {len(history)-1:12d}  "
                    f"{status}\n"
                )

            file.write("\n")

            plot_broken_lines(bx, by, bhistory, bxs, bys, eps)
            plot_newton(results, eps)

        file.write("=" * 85 + "\n")
        file.write("Производные функции:\n")
        file.write("f'(x) = 1 + x/5 + cos(x)\n")
        file.write("f''(x) = 0.2 - sin(x)\n")
        file.write("=" * 85 + "\n")


# ============================================================
# ЗАПУСК
# ============================================================

if __name__ == "__main__":
    os.makedirs(GRAPH_DIR, exist_ok=True)

    write_results()

    print("Вычисления завершены.")
    print(f"Текстовые результаты: {RESULT_FILE}")
    print(f"Графики сохранены в папке: {GRAPH_DIR}")
    print("Графики не открывались на экране.")