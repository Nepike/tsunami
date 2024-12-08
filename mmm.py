import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Параметры системы
g = 9.81  # Ускорение свободного падения
D0 = 50  # Базовая глубина
hill_height = 30
hill_width = 20
hill_x, hill_y = 50, 50  # Центр подводной структуры
L = 100  # Размер области


# Определение функции глубины
def depth_profile(x, y, profile_type):
    if profile_type == "Гора":
        return D0 - hill_height * np.exp(-((x - hill_x) ** 2 + (y - hill_y) ** 2) / (2 * hill_width ** 2))
    elif profile_type == "Впадина":
        return D0 + hill_height * np.exp(-((x - hill_x) ** 2 + (y - hill_y) ** 2) / (2 * hill_width ** 2))
    elif profile_type == "Хребет":
        return D0 - hill_height * np.exp(-(x - hill_x + L / 5) ** 2 / (2 * hill_width ** 2))
    elif profile_type == "Плато":
        return D0 - hill_height * (x / L)
    elif profile_type == "Случайный":
        np.random.seed(0)
        return D0 - hill_height * np.random.rand(*x.shape)
    elif profile_type == "Многослойный":
        layer_width = L / 10  # Ширина одного слоя
        return D0 - hill_height * (np.floor(x / layer_width) % 2)


# Градиенты глубины
def depth_gradients(x, y, profile_type):
    if profile_type == "Гора" or profile_type == "Впадина":
        grad_x = -hill_height * (x - hill_x) * np.exp(
            -((x - hill_x) ** 2 + (y - hill_y) ** 2) / (2 * hill_width ** 2)) / hill_width ** 2
        grad_y = -hill_height * (y - hill_y) * np.exp(
            -((x - hill_x) ** 2 + (y - hill_y) ** 2) / (2 * hill_width ** 2)) / hill_width ** 2
    elif profile_type == "Хребет":
        grad_x = hill_height * (x - hill_x + L / 5) * np.exp(
            -(x - hill_x + L / 5) ** 2 / (2 * hill_width ** 2)) / hill_width ** 2
        grad_y = 0
    elif profile_type == "Плато":
        grad_x = hill_height / L
        grad_y = 0
    else:
        grad_x = grad_y = 0  # Для случайного и многослойного профиля градиенты не определены
    return -grad_x, -grad_y



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


# Параметры для анимации
profile_type = "Плато"#  "Гора", "Впадина", "Хребет", "Плато", "Случайный", "Многослойный"
num_directions = 40
radius = 20  # Величина начального импульса
x0, y0 = 20, 20  # Начальная точка

angles = np.linspace(0, 2 * np.pi, num_directions, endpoint=False)
initial_conditions = [[x0, y0, radius * np.cos(angle), radius * np.sin(angle)] for angle in angles]

t_span = (0, 20)  # Временной интервал
t_eval = np.linspace(t_span[0], t_span[1], 500)

# Решение уравнений Гамильтона
trajectories = []
for cond in initial_conditions:
    sol = solve_ivp(hamiltonian_system, t_span, cond, t_eval=t_eval, args=(profile_type,), method='RK45')
    trajectories.append(sol.y)

# Настройка анимации
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_title(f"Анимация траекторий ({profile_type})")
ax.set_xlabel("x")
ax.set_ylabel("y")

lines = [ax.plot([], [], lw=2)[0] for _ in initial_conditions]


def update(frame):
    for line, trajectory in zip(lines, trajectories):
        line.set_data(trajectory[0, :frame], trajectory[1, :frame])
    return lines


ani = FuncAnimation(fig, update, frames=len(t_eval), interval=50, blit=True)

# Показываем анимацию
plt.show()
