import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import tkinter as tk
from tkinter import ttk

# Параметры системы
L = 100  # Размер области
g = 9.81  # Ускорение свободного падения
D0 = 50  # Базовая глубина
hill_height = 40
hill_width = 20
hill_x, hill_y = 50 / 100 * L, 50 / 100 * L  # Центр подводной структуры


# Определение функции глубины
def depth_profile(x, y, profile_type):
    if profile_type == "Гора":
        return D0 - hill_height * np.exp(-((x - hill_x) ** 2 + (y - hill_y) ** 2) / (2 * hill_width ** 2))
    elif profile_type == "Впадина":
        return D0 + hill_height * np.exp(-((x - hill_x) ** 2 + (y - hill_y) ** 2) / (2 * hill_width ** 2))
    elif profile_type == "Хребет":
        return D0 - hill_height * np.exp(-(x - hill_x) ** 2 / (2 * hill_width ** 2))
    elif profile_type == "Плато":
        return D0 - hill_height * (x / L) + 1
    elif profile_type == "Случайный":
        np.random.seed(0)
        return D0 - hill_height * np.random.rand(*x.shape)
    elif profile_type == "Многослойный":
        layer_width = L / 10  # Ширина одного слоя
        return D0 - hill_height * (np.floor(x / layer_width) % 2)


# Градиенты глубины
def depth_gradients(x, y, profile_type):
    if profile_type == "Гора":
        grad_x = -hill_height * (x - hill_x) * np.exp(
            -((x - hill_x) ** 2 + (y - hill_y) ** 2) / (2 * hill_width ** 2)) / hill_width ** 2
        grad_y = -hill_height * (y - hill_y) * np.exp(
            -((x - hill_x) ** 2 + (y - hill_y) ** 2) / (2 * hill_width ** 2)) / hill_width ** 2
    elif profile_type == "Впадина":
        grad_x = hill_height * (x - hill_x) * np.exp(
            -((x - hill_x) ** 2 + (y - hill_y) ** 2) / (2 * hill_width ** 2)) / hill_width ** 2
        grad_y = hill_height * (y - hill_y) * np.exp(
            -((x - hill_x) ** 2 + (y - hill_y) ** 2) / (2 * hill_width ** 2)) / hill_width ** 2
    elif profile_type == "Хребет":
        grad_x = -hill_height * (x - hill_x) * np.exp(-(x - hill_x) ** 2 / (2 * hill_width ** 2)) / hill_width ** 2
        grad_y = 0
    elif profile_type == "Плато":
        grad_x = hill_height / L
        grad_y = 0
    else:
        grad_x = grad_y = 0  # Для случайного и многослойного профиля градиенты не определены
    return -grad_x, -grad_y


# Скорость волны
def wave_speed(x, y, profile_type):
    return np.sqrt(g * depth_profile(x, y, profile_type))


# Уравнения Гамильтона
def hamiltonian_system(t, z, profile_type):
    x, y, px, py = z
    c = wave_speed(x, y, profile_type)

    dxdt = c * px / np.sqrt(px ** 2 + py ** 2)  # x' = ∂H/∂px
    dydt = c * py / np.sqrt(px ** 2 + py ** 2)  # y' = ∂H/∂py

    grad_D_x, grad_D_y = depth_gradients(x, y, profile_type)
    dp_x = -0.5 * g * grad_D_x  # px' = -∂H/∂x
    dp_y = -0.5 * g * grad_D_y  # py' = -∂H/∂y

    return [dxdt, dydt, dp_x, dp_y]


# Функция запуска симуляции
def run_simulation():
    global hill_height, hill_width, hill_x, hill_y

    hill_height = float(hill_height_var.get())
    hill_width = float(hill_width_var.get())
    hill_x = float(hill_x_var.get()) / 100 * L
    hill_y = float(hill_y_var.get()) / 100 * L

    x0 = float(x0_var.get()) / 100 * L
    y0 = float(y0_var.get()) / 100 * L
    radius = float(radius_var.get())
    profile_type = profile_var.get()

    save_animation = save_var.get()

    num_directions = 160
    angles = np.linspace(0, 2 * np.pi, num_directions, endpoint=False)
    initial_conditions = [[x0, y0, radius * np.cos(angle), radius * np.sin(angle)] for angle in angles]

    t_span = (0, 20)  # Временной интервал
    t_eval = np.linspace(t_span[0], t_span[1], 500)

    # Решение уравнений Гамильтона
    trajectories = []
    for cond in initial_conditions:
        sol = solve_ivp(hamiltonian_system, t_span, cond, t_eval=t_eval, args=(profile_type,), method='RK45')
        trajectories.append(sol.y)

    # Создание сетки для профиля глубины
    x = np.linspace(0, L, 200)
    y = np.linspace(0, L, 200)
    X, Y = np.meshgrid(x, y)
    Z = D0 - depth_profile(X, Y, profile_type)

    # Настройка анимации
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = ax.imshow(Z, extent=(0, L, 0, L), origin="lower", cmap="viridis", alpha=0.5)
    fig.colorbar(cmap, ax=ax, label="Высота (м)")

    initial_conditions_str = (
                              f"Импульс x0={x0:.1f}, y0={y0:.1f}\n"
                              f"Импульс v={radius:.1f}\n"
                              f"Профиль дна={profile_type}\n"
                              f"Структура x={hill_x:.1f}, y={hill_y:.1f}\n"
                              f"Структура w={hill_width}, h={hill_height}")
    ax.set_title(initial_conditions_str)

    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Анимация траекторий
    lines = [ax.plot([], [], lw=2)[0] for _ in initial_conditions]

    def update(frame):
        for line, trajectory in zip(lines, trajectories):
            line.set_data(trajectory[0, :frame], trajectory[1, :frame])
        return lines

    ani = FuncAnimation(fig, update, frames=len(t_eval), interval=50, blit=True)
    if save_animation:
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        filename = f"ham_tsunami_wave_{timestamp}.gif"
        ani.save(filename, writer="imagemagick", fps=25)

    plt.show()


# Интерфейс tkinter
root = tk.Tk()
root.title("Настройка параметров симуляции")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Ввод параметров
x0_var = tk.StringVar(value="20")
y0_var = tk.StringVar(value="20")
radius_var = tk.StringVar(value="15")
hill_height_var = tk.StringVar(value="40")
hill_width_var = tk.StringVar(value="20")
hill_x_var = tk.StringVar(value="50")
hill_y_var = tk.StringVar(value="50")
profile_var = tk.StringVar(value="Гора")
save_var = tk.BooleanVar(value=False)

# Метки и поля ввода
# Начальные условия
ttk.Label(frame, text="Начальное x0 (%):").grid(row=0, column=0, padx=5, pady=5)
ttk.Entry(frame, textvariable=x0_var).grid(row=0, column=1, padx=5, pady=5)

ttk.Label(frame, text="Начальное y0 (%):").grid(row=1, column=0, padx=5, pady=5)
ttk.Entry(frame, textvariable=y0_var).grid(row=1, column=1, padx=5, pady=5)

# Радиус импульса
ttk.Label(frame, text="Радиус импульса:").grid(row=2, column=0, padx=5, pady=5)
ttk.Entry(frame, textvariable=radius_var).grid(row=2, column=1, padx=5, pady=5)

# Параметры горы
ttk.Label(frame, text="Высота горы:").grid(row=3, column=0, padx=5, pady=5)
ttk.Entry(frame, textvariable=hill_height_var).grid(row=3, column=1, padx=5, pady=5)

ttk.Label(frame, text="Ширина горы:").grid(row=4, column=0, padx=5, pady=5)
ttk.Entry(frame, textvariable=hill_width_var).grid(row=4, column=1, padx=5, pady=5)

ttk.Label(frame, text="Координата горы x (%):").grid(row=5, column=0, padx=5, pady=5)
ttk.Entry(frame, textvariable=hill_x_var).grid(row=5, column=1, padx=5, pady=5)

ttk.Label(frame, text="Координата горы y (%):").grid(row=6, column=0, padx=5, pady=5)
ttk.Entry(frame, textvariable=hill_y_var).grid(row=6, column=1, padx=5, pady=5)

# Тип профиля
ttk.Label(frame, text="Тип профиля:").grid(row=7, column=0, padx=5, pady=5)
ttk.Combobox(frame, textvariable=profile_var,
             values=["Гора", "Впадина", "Хребет", "Плато", "Случайный", "Многослойный"], state="readonly").grid(row=7,
                                                                                                                column=1,
                                                                                                                padx=5,
                                                                                                                pady=5)

# Чекбокс для сохранения анимации
ttk.Checkbutton(frame, text="Сохранить анимацию", variable=save_var).grid(row=8, column=0, columnspan=2, pady=5)

# Кнопка запуска
ttk.Button(frame, text="Запустить", command=run_simulation).grid(row=9, column=0, columnspan=2, pady=10)

root.mainloop()
