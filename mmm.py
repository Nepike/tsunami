import datetime
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
        grad_x = +hill_height * (x - hill_x) * np.exp(
            -((x - hill_x) ** 2 + (y - hill_y) ** 2) / (2 * hill_width ** 2)) / hill_width ** 2
        grad_y = +hill_height * (y - hill_y) * np.exp(
            -((x - hill_x) ** 2 + (y - hill_y) ** 2) / (2 * hill_width ** 2)) / hill_width ** 2
    elif profile_type == "Хребет":
        grad_x = hill_height * (x - hill_x ) * np.exp(
            -(x - hill_x) ** 2 / (2 * hill_width ** 2)) / hill_width ** 2
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
profile_type = "Плато"  # "Гора", "Впадина", "Хребет", "Плато", "Случайный", "Многослойный"
num_directions = 160
radius = 20  # Величина начального импульса
x0, y0 = 10, 10  # Начальная точка

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
Z = depth_profile(X, Y, profile_type)

# Настройка анимации
fig, ax = plt.subplots(figsize=(6, 6))
# Отображение цветной карты профиля глубины
cmap = ax.imshow(Z, extent=(0, L, 0, L), origin="lower", cmap="viridis", alpha=0.5)
fig.colorbar(cmap, ax=ax, label="Глубина (м)")

ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_title(f"Анимация траекторий ({profile_type})")
ax.set_xlabel("x")
ax.set_ylabel("y")

lines = [ax.plot([], [], lw=2)[0] for _ in initial_conditions]

def update(frame):
    for line, trajectory in zip(lines, trajectories):
        line.set_data(trajectory[0, :frame], trajectory[1, :frame])
    return lines

ani = FuncAnimation(fig, update, frames=len(t_eval), interval=50, blit=True)
now = datetime.datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")
filename = f"ham_tsunami_wave_{timestamp}.gif"
ani.save(filename, writer="imagemagick", fps=25)

# Показываем анимацию
plt.show()
