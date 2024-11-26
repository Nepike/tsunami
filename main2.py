import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Параметры среды
L = 400  # Размер области
dx = 5  # Шаг сетки по пространству
nx = int(L / dx)
x = np.linspace(0, L, nx)
y = np.linspace(0, L, nx)
X, Y = np.meshgrid(x, y)

# Параметры времени
dt = 0.05  # Шаг по времени
t_steps = 1000  # Количество временных шагов

# Профиль глубины (плоский или с горой)
D0 = 20  # Средняя глубина
hill_height = -40  # Высота горы
hill_width = 20  # Ширина горы
hill_center_x = L / 3
hill_center_y = L / 2


def depth_profile(x, y):
    #return D0 - hill_height * np.array([np.sin( xi/ 60) if np.sin( xi/ 60) > 0 else 0 for xi in x]) * np.cos(y/60)
    return D0 - hill_height * np.exp(-((x - hill_center_x) ** 2 + (y - hill_center_y) ** 2) / (2 * hill_width ** 2))


D = depth_profile(X, Y)
g = 9.81
c = 1*np.sqrt(g * D)  # Волновая скорость зависит от глубины

# Начальные условия - локализованный импульс
initial_height = 60
initial_center_x = L / 8
initial_center_y = L / 8
initial_width = 15


def initial_wave(x, y):
    return initial_height * np.exp(-((x - initial_center_x) ** 2 + (y - initial_center_y) ** 2) / (2 * initial_width ** 2))


# Начальные массивы для высоты волны η и её изменения во времени
eta = D0 + initial_wave(X, Y)  # Гладь воды на уровне D0 плюс начальное возмущение
eta_prev = eta.copy()  # Высота на предыдущем временном шаге
eta_next = np.zeros_like(eta)  # Шаблон для следующего шага

# Анимация
fig, ax = plt.subplots()
# fig.set_figwidth(10)
# fig.set_figheight(7)
ax.set_xlim(0, L)
ax.set_ylim(0, L)
cax = ax.imshow(eta, cmap="viridis", vmin=0, vmax=initial_height+D0, origin="lower", extent=[0, L, 0, L])
fig.colorbar(cax, ax=ax)


def update(frame):
    global eta, eta_prev, eta_next, wave_surf, depth_surf

    # Разностная схема для волнового уравнения
    eta_next[1:-1, 1:-1] = (2 * eta[1:-1, 1:-1] - eta_prev[1:-1, 1:-1] +
                            (dt ** 2 / dx ** 2) * (
                                c[1:-1, 1:-1] ** 2 * (
                                    (eta[2:, 1:-1] - 2 * eta[1:-1, 1:-1] + eta[:-2, 1:-1]) +
                                    (eta[1:-1, 2:] - 2 * eta[1:-1, 1:-1] + eta[1:-1, :-2])
                                )
                            ))

    # Условия Неймана (нулевой градиент на границах)
    eta_next[0, 1:-1] = eta_next[1, 1:-1]  # Верхняя граница
    eta_next[-1, 1:-1] = eta_next[-2, 1:-1]  # Нижняя граница
    eta_next[1:-1, 0] = eta_next[1:-1, 1]  # Левая граница
    eta_next[1:-1, -1] = eta_next[1:-1, -2]  # Правая граница

    # Угловые точки (обработка с использованием ближайших соседей)
    eta_next[0, 0] = eta_next[1, 1]  # Верхний левый угол
    eta_next[0, -1] = eta_next[1, -2]  # Верхний правый угол
    eta_next[-1, 0] = eta_next[-2, 1]  # Нижний левый угол
    eta_next[-1, -1] = eta_next[-2, -2]  # Нижний правый угол

    # Обновляем массивы для следующего временного шага
    eta_prev, eta = eta, eta_next.copy()


    cax.set_array(eta)
    return cax,


ani = FuncAnimation(fig, update, frames=t_steps, interval=50)
#ani.save("tsunami_wave.gif", writer="imagemagick", fps=15)


plt.show()
