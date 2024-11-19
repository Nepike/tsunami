import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Параметры среды
L = 500  # Размер области
dx = 5  # Шаг сетки по пространству
nx = int(L / dx)
x = np.linspace(0, L, nx)
y = np.linspace(0, L, nx)
X, Y = np.meshgrid(x, y)

# Параметры времени
dt = 0.05  # Шаг по времени
t_steps = 200  # Количество временных шагов

# Профиль глубины (подводная гора)
hill_height = 20  # Высота горы
hill_width = 50  # Ширина горы
hill_center_x = L / 2
hill_center_y = L / 2
D0 = 20  # Средняя глубина (без горы)


def depth_profile(x, y):
    return D0 - hill_height * np.exp(-((x - hill_center_x) ** 2 + (y - hill_center_y) ** 2) / (2 * hill_width ** 2))


D = depth_profile(X, Y)
g = 9.81
c = 3*np.sqrt(g * D)  # Скорость распространения волны в зависимости от глубины

# Начальные условия - локализованный импульс
initial_height = 10
initial_center_x = L / 8
initial_center_y = L / 8
initial_width = 20


def initial_wave(x, y):
    return initial_height * np.exp(
        -((x - initial_center_x) ** 2 + (y - initial_center_y) ** 2) / (2 * initial_width ** 2))


# Начальные массивы для высоты волны η и её изменения во времени
eta = initial_wave(X, Y)
eta_prev = eta.copy()  # Начальная высота
eta_next = np.zeros_like(eta)

# Настройка 3D графика
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(-initial_height, initial_height)

surf = ax.plot_surface(X, Y, eta, cmap="viridis", edgecolor="k")


def update(frame):
    global eta, eta_prev, eta_next

    # Разностная схема для волнового уравнения
    for i in range(1, nx - 1):
        for j in range(1, nx - 1):
            # Используем разностную схему для вычисления высоты волны в следующем временном шаге
            eta_next[i, j] = (2 * eta[i, j] - eta_prev[i, j] +
                              (dt ** 2 * c[i, j] ** 2 / dx ** 2) * (
                                      (eta[i + 1, j] - 2 * eta[i, j] + eta[i - 1, j]) +
                                      (eta[i, j + 1] - 2 * eta[i, j] + eta[i, j - 1])
                              ))

    # Условия полного отражения: высота на границах остается прежней
    eta_next[0, :] = eta[0, :]
    eta_next[-1, :] = eta[-1, :]
    eta_next[:, 0] = eta[:, 0]
    eta_next[:, -1] = eta[:, -1]

    # Обновляем массивы для следующего временного шага
    eta_prev, eta, eta_next = eta, eta_next, eta_prev

    # Обновление графика
    ax.clear()
    ax.set_zlim(-initial_height, initial_height)
    ax.plot_surface(X, Y, eta, cmap="viridis", edgecolor="k")
    ax.set_title(f"Time Step: {frame}")
    return ax,


ani = FuncAnimation(fig, update, frames=t_steps, interval=50)
plt.show()
