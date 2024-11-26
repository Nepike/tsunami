import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D


# Параметры среды
L = 500  # Размер области
dx = 5  # Шаг сетки
nx = int(L / dx)
x = np.linspace(0, L, nx)
y = np.linspace(0, L, nx)
X, Y = np.meshgrid(x, y)

# Профиль глубины
hill_height = 5
hill_width = 50
hill_center_x = L / 2
hill_center_y = L / 2
D0 = 10  # Средняя глубина

def depth_profile(x, y):
    return D0 + hill_height * np.exp(-((x - hill_center_x)**2 + (y - hill_center_y)**2) / (2 * hill_width**2))

D = depth_profile(X, Y)
g = 9.81
c = np.sqrt(g * D)

# Начальные условия
initial_height = 10
initial_center_x = L / 8
initial_center_y = L / 8
initial_width = 50

def initial_wave(x, y):
    return initial_height * np.exp(-((x - initial_center_x)**2 + (y - initial_center_y)**2) / (2 * initial_width**2))

# Начальные условия для волнового поля
Z = initial_wave(X, Y)
u0 = Z.flatten()  # начальная высота
u1 = np.zeros_like(u0)  # начальная скорость (нулевая)

# Гамильтониан
def hamiltonian(t, u):
    x, y, px, py = u
    # Вычисляем скорость c(x, y) в зависимости от глубины
    c_local = np.sqrt(g * depth_profile(x, y))
    # Уравнения Гамильтона для d(x, y)/dt = dH/d(px, py) и d(px, py)/dt = -dH/d(x, y)
    dxdt = c_local * px / np.sqrt(px**2 + py**2)
    dydt = c_local * py / np.sqrt(px**2 + py**2)
    dp_xdt = -0.5 * g * hill_height * np.exp(-((x - hill_center_x)**2 + (y - hill_center_y)**2) / (2 * hill_width**2)) * (x - hill_center_x) / (hill_width**2)
    dp_ydt = -0.5 * g * hill_height * np.exp(-((x - hill_center_x)**2 + (y - hill_center_y)**2) / (2 * hill_width**2)) * (y - hill_center_y) / (hill_width**2)
    return [dxdt, dydt, dp_xdt, dp_ydt]

# Начальные условия для траекторий волнового фронта
x0, y0 = initial_center_x, initial_center_y
px0, py0 = 0, 0  # начальные импульсы (направление движения волнового фронта)

# Решаем уравнения Гамильтона
sol = solve_ivp(hamiltonian, [0, 100], [x0, y0, px0, py0], t_eval=np.linspace(0, 100, 500))

# Анимация волнового фронта
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_zlim(-initial_height, initial_height)

wave = [initial_wave(X, Y)]

def update(frame):
    ax.clear()
    # Обновляем начальные условия волнового фронта на основе траекторий
    X_wave = sol.y[0, frame]
    Y_wave = sol.y[1, frame]
    wave_front = initial_wave(X - X_wave, Y - Y_wave)  # смещаем волну в соответствии с фронтом
    wave[0] = wave_front
    ax.plot_surface(X, Y, wave_front, cmap="viridis", edgecolor="none")
    return ax

ani = FuncAnimation(fig, update, frames=500, interval=50)
ani.save("tsunami_wave.gif", writer="imagemagick", fps=10)

plt.show()
