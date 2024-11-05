import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Параметры среды
L = 500  # Размер области
dx = 5  # Шаг сетки
nx = int(L / dx)
x = np.linspace(0, L, nx)
y = np.linspace(0, L, nx)
X, Y = np.meshgrid(x, y)

# Параметры волны
g = 9.8  # Ускорение свободного падения
D0 = 50  # Средняя глубина

# Профиль глубины
hill_height = -20  # Высота подводной горы
hill_width = 50
hill_center = (L / 2, L / 2)

def depth_profile(x, y):
    hill = hill_height * np.exp(-(((x - hill_center[0]) ** 2 + (y - hill_center[1]) ** 2) / (2 * hill_width ** 2)))
    shore = 20 * (L - np.maximum(x, y)) / L  # постепенный подъём берега
    return np.maximum(D0 - hill + shore, 1)

D = depth_profile(X, Y)
c = np.sqrt(g * D)  # Скорость волны

# Начальные условия - круговой волновой импульс
A = 15  # Амплитуда волны
sigma = 40.0  # Ширина волны
x0, y0 = L / 4, L / 4

def initial_wave(x, y):
    return A * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

Z = initial_wave(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(-hill_height, A + D0)
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Height")

wave_surf = ax.plot_surface(X, Y, Z + D, cmap='viridis', zorder=1)
hill_surf = ax.plot_surface(X, Y, depth_profile(X, Y), color='brown', alpha=0.5, zorder=0)

def update_wave(t):
    wave = A * np.exp(-((X - x0 - c * t) ** 2 + (Y - y0 - c * t) ** 2) / (2 * sigma ** 2))
    return wave + D

def animate(i):
    t = i * 0.1
    Z_new = update_wave(t)
    ax.clear()
    ax.set_zlim(-hill_height, A + D0)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Height")
    ax.set_title(f"Time: {t:.2f} s")
    ax.plot_surface(X, Y, Z_new, cmap='viridis', edgecolor='none')
    return ax

ani = FuncAnimation(fig, animate, frames=300, interval=50, blit=False)

# Сохранение анимации как GIF
print("Сохраняем анимацию как GIF...")
ani.save("tsunami_wave_3D.gif", writer="imagemagick", fps=10)
print("Гифка сохранена как 'tsunami_wave_3D.gif'.")

plt.show()
