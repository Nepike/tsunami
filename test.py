import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk
import datetime


sources = []

def add_source():
    multiplier = float(multiplier_var.get())
    L = int(length_var.get()) * multiplier

    x = float(source_x_var.get()) * L / 100
    y = float(source_y_var.get()) * L / 100
    height = float(source_height_var.get()) * multiplier
    width = float(source_width_var.get()) * multiplier
    delay = float(source_delay_var.get())  # Добавить задержку
    sources.append({"x": x, "y": y, "height": height, "width": width, "delay": delay})
    sources_listbox.insert(
        tk.END,
        f"Источник: x={x:.2f}, y={y:.2f}, h={height}, w={width}, delay={delay}s"
    )


def clear_sources():
    sources.clear()
    sources_listbox.delete(0, tk.END)

def run_simulation():
    multiplier = float(multiplier_var.get())
    speed_multiplier = float(speed_multiplier_var.get())

    L = int(length_var.get())*multiplier
    D0 = float(depth_var.get())*multiplier

    hill_height = float(hill_height_var.get())*multiplier
    hill_width = float(hill_width_var.get())*multiplier
    hill_x = L*float(hill_x_var.get())/100
    hill_y = L*float(hill_y_var.get())/100

    for source in sources:
        source['used'] = False



    # Пространственные параметры
    dx = 5
    nx = int(L / dx)
    x = np.linspace(0, L, nx)
    y = np.linspace(0, L, nx)
    X, Y = np.meshgrid(x, y)

    # Параметры времени
    dt = 0.05  # Шаг по времени
    t_steps = int(gif_time_var.get())  # Количество временных шагов

    # Выбор профиля глубины
    profile_type = depth_profile_var.get()  # Тип профиля

    def depth_profile(x, y, profile_type):
        if profile_type == "Гора":
            return D0 - hill_height * np.exp(-((x - hill_x) ** 2 + (y - hill_y) ** 2) / (2 * hill_width ** 2))
        elif profile_type == "Впадина":
            return D0 + hill_height * np.exp(-((x - hill_x) ** 2 + (y - hill_y) ** 2) / (2 * hill_width ** 2))
        elif profile_type == "Хребет":
            return D0 - hill_height * np.exp(-(x - hill_x) ** 2 / (2 * hill_width ** 2))
        elif profile_type == "Плато":
            return D0 - hill_height * (x / L)
        elif profile_type == "Случайный":
            return D0 - hill_height * np.random.rand(*x.shape)
        elif profile_type == "Многослойный":
            layer_width = L / 4  # Ширина одного слоя
            return D0 - hill_height * (np.floor(x / layer_width) % 2)
        else:
            raise ValueError("Неизвестный тип профиля!")


    D = depth_profile(X, Y, profile_type)
    g = 9.81
    c = speed_multiplier*np.sqrt(g * D)  # Волновая скорость зависит от глубины

    def initial_wave(x, y, sources, time):
        """Создание волны с учётом временной задержки."""
        wave = np.zeros_like(x)
        for source in sources:
            if time >= source["delay"] and not source["used"]:  # Проверяем, активен ли источник
                wave += source["height"] * np.exp(
                    -((x - source["x"]) ** 2 + (y - source["y"]) ** 2) / (2 * source["width"] ** 2)
                )
                source["used"] = True
        return wave

    eta = D0 + initial_wave(X, Y, sources, 0)
    eta_prev = eta.copy()
    eta_next = np.zeros_like(eta)

    # Настройка графиков
    fig = plt.figure(figsize=(18, 8)) #10 5

    # 2D график
    ax2d = fig.add_subplot(121)
    ax2d.set_xlim(0, L)
    ax2d.set_ylim(0, L)
    cax2d = ax2d.imshow(eta, cmap="viridis", vmin=0, vmax=D0 + max([source['height'] for source in sources]+[0]), origin="lower", extent=[0, L, 0, L])
    fig.colorbar(cax2d, ax=ax2d, label="Высота волны")

    # 3D график
    ax3d = fig.add_subplot(122, projection='3d')
    ax3d.set_zlim(0, D0 + max([source['height'] for source in sources]+[0]))
    wave_surf = ax3d.plot_surface(X, Y, eta, cmap="viridis", edgecolor="k", alpha=0.8)
    depth_surf = ax3d.plot_surface(X, Y, -D + D0, cmap="copper", edgecolor="none", alpha=0.6)

    def update(frame):
        nonlocal eta, eta_prev, eta_next, wave_surf
        global sources


        # Разностная схема для волнового уравнения
        eta_next[1:-1, 1:-1] = (2 * eta[1:-1, 1:-1] - eta_prev[1:-1, 1:-1] +
                                (dt**2 / dx**2) * (
                                    c[1:-1, 1:-1]**2 * (
                                        (eta[2:, 1:-1] - 2 * eta[1:-1, 1:-1] + eta[:-2, 1:-1]) +
                                        (eta[1:-1, 2:] - 2 * eta[1:-1, 1:-1] + eta[1:-1, :-2])
                                    )
                                ))
        time = frame * dt
        for source in sources:
            if not source["used"] and time >= source["delay"]:
                wave = initial_wave(X, Y, sources, time)
                eta_next += wave
                eta += wave

        # Условия Неймана (нулевой градиент на границах)
        eta_next[0, 1:-1] = eta_next[1, 1:-1]  # Верхняя граница
        eta_next[-1, 1:-1] = eta_next[-2, 1:-1]  # Нижняя граница
        eta_next[1:-1, 0] = eta_next[1:-1, 1]  # Левая граница
        eta_next[1:-1, -1] = eta_next[1:-1, -2]  # Правая граница
        eta_next[0, 0] = eta_next[1, 1]  # Верхний левый угол
        eta_next[0, -1] = eta_next[1, -2]  # Верхний правый угол
        eta_next[-1, 0] = eta_next[-2, 1]  # Нижний левый угол
        eta_next[-1, -1] = eta_next[-2, -2]  # Нижний правый угол

        eta_prev, eta = eta, eta_next.copy()

        cax2d.set_array(eta)

        wave_surf.remove()
        wave_surf = ax3d.plot_surface(X, Y, eta, cmap="viridis", edgecolor="k", alpha=0.8)

        return cax2d, wave_surf, depth_surf

    ani = FuncAnimation(fig, update, frames=t_steps, interval=30)
    if bool(save_gif_var.get()):
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        filename = f"tsunami_wave_{timestamp}.gif"
        ani.save(filename, writer="imagemagick", fps=25)
        print(f"{timestamp} finished!")
        return
    else:
        plt.show()


root = tk.Tk()
root.title("Настройка параметров симуляции")

depth_profile_var = tk.StringVar(value="Гора")

length_var = tk.StringVar(value="300")
depth_var = tk.StringVar(value="20")

hill_height_var = tk.StringVar(value="19")
hill_width_var = tk.StringVar(value="20")
hill_x_var = tk.StringVar(value="80")
hill_y_var = tk.StringVar(value="30")

multiplier_var = tk.StringVar(value="1")
speed_multiplier_var = tk.StringVar(value="1")
save_gif_var = tk.BooleanVar(value=False)
gif_time_var = tk.StringVar(value="1000")

ttk.Label(root, text="Тип профиля глубины:").grid(row=0, column=2, padx=5, pady=5)
ttk.OptionMenu(root, depth_profile_var, "Гора", "Гора", "Впадина", "Хребет", "Плато", "Случайный", "Многослойный").grid(row=0, column=3, padx=5, pady=5)

ttk.Label(root, text="Размер области (L):").grid(row=0, column=0, padx=5, pady=5)
ttk.Entry(root, textvariable=length_var).grid(row=0, column=1, padx=5, pady=5)

ttk.Label(root, text="Средняя глубина (D0):").grid(row=1, column=0, padx=5, pady=5)
ttk.Entry(root, textvariable=depth_var).grid(row=1, column=1, padx=5, pady=5)

ttk.Label(root, text="Высота горы:").grid(row=2, column=0, padx=5, pady=5)
ttk.Entry(root, textvariable=hill_height_var).grid(row=2, column=1, padx=5, pady=5)

ttk.Label(root, text="Ширина горы:").grid(row=3, column=0, padx=5, pady=5)
ttk.Entry(root, textvariable=hill_width_var).grid(row=3, column=1, padx=5, pady=5)

ttk.Label(root, text="X% горы:").grid(row=4, column=0, padx=5, pady=5)
ttk.Entry(root, textvariable=hill_x_var).grid(row=4, column=1, padx=5, pady=5)

ttk.Label(root, text="Y% Горы:").grid(row=5, column=0, padx=5, pady=5)
ttk.Entry(root, textvariable=hill_y_var).grid(row=5, column=1, padx=5, pady=5)


ttk.Label(root, text="Множитель:").grid(row=6, column=0, padx=5, pady=5)
ttk.Entry(root, textvariable=multiplier_var).grid(row=6, column=1, padx=5, pady=5)

ttk.Label(root, text="Множитель скорости:").grid(row=7, column=0, padx=5, pady=5)
ttk.Entry(root, textvariable=speed_multiplier_var).grid(row=7, column=1, padx=5, pady=5)

ttk.Label(root, text="Записать гифку?").grid(row=8, column=0, padx=5, pady=5)
ttk.Checkbutton(root, variable=save_gif_var).grid(row=8, column=1, padx=5, pady=5)

ttk.Label(root, text="Промежуток времени:").grid(row=9, column=0, padx=5, pady=5)
ttk.Entry(root, textvariable=gif_time_var).grid(row=9, column=1, padx=5, pady=5)

ttk.Button(root, text="Рассчитать анимацию", command=run_simulation).grid(row=9, column=3, columnspan=3, pady=10)


ttk.Label(root, text="X% источника:").grid(row=1, column=2, padx=5, pady=5)
source_x_var = tk.StringVar(value="50")
ttk.Entry(root, textvariable=source_x_var).grid(row=1, column=3, padx=5, pady=5)

ttk.Label(root, text="Y% источника:").grid(row=2, column=2, padx=5, pady=5)
source_y_var = tk.StringVar(value="50")
ttk.Entry(root, textvariable=source_y_var).grid(row=2, column=3, padx=5, pady=5)

ttk.Label(root, text="Высота источника:").grid(row=3, column=2, padx=5, pady=5)
source_height_var = tk.StringVar(value="40")
ttk.Entry(root, textvariable=source_height_var).grid(row=3, column=3, padx=5, pady=5)

ttk.Label(root, text="Ширина источника:").grid(row=4, column=2, padx=5, pady=5)
source_width_var = tk.StringVar(value="10")
ttk.Entry(root, textvariable=source_width_var).grid(row=4, column=3, padx=5, pady=5)

ttk.Label(root, text="Задержка источника (сек):").grid(row=5, column=2, padx=5, pady=5)
source_delay_var = tk.StringVar(value="0")
ttk.Entry(root, textvariable=source_delay_var).grid(row=5, column=3, padx=5, pady=5)


ttk.Button(root, text="Добавить источник", command=add_source).grid(row=4, column=4, columnspan=1, pady=5)
ttk.Button(root, text="Очистить источники", command=clear_sources).grid(row=6, column=4, columnspan=1, pady=5)

ttk.Label(root, text="Источники:").grid(row=7, column=2, padx=5, pady=5)
sources_listbox = tk.Listbox(root, height=10, width=55)
sources_listbox.grid(row=7, column=3, columnspan=2, padx=5, pady=15)


root.mainloop()
