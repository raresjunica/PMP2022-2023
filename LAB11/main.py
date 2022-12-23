import numpy as np
import arviz as az
import random
import pymc3 as pm
import matplotlib.pyplot as plt
import scipy.stats as stats


# 1

def posterior_grid(grid_points=50, heads=6, tails=9):
    """
    A grid implementation for the coin-flipping problem
    """
    grid = np.linspace(0, 1, grid_points)
    prior = (grid <= 0.5).astype(int)
    likelihood = stats.binom.pmf(heads, heads + tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior


data = np.repeat([0, 1], (5, 40))
points = 10
h = data.sum()
t = len(data) - h
grid, posterior = posterior_grid(points, h, t)
plt.plot(grid, posterior, 'o-')
plt.title(f'heads = {h}, tails = {t}')
plt.yticks([])
plt.xlabel('θ')
plt.show()

# 2

N = 100
x, y = np.random.uniform(-1, 1, size=(2, N))
inside = (x * 2 + y * 2) <= 1
pi = inside.sum() * 4 / N
error = abs((pi - np.pi) / pi) * 100
outside = np.invert(inside)
plt.figure(figsize=(8, 8))
plt.plot(x[inside], y[inside], 'b.')
plt.plot(x[outside], y[outside], 'r.')
plt.plot(0, 0, label=f'π*= {pi:4.3f}error = {error:4.3f}', alpha=0)
plt.axis('square')
plt.xticks([])
plt.yticks([])
plt.legend(loc=1, frameon=True, framealpha=0.9)
plt.show()

N = 1000
x, y = np.random.uniform(-1, 1, size=(2, N))
inside = (x * 2 + y * 2) <= 1
pi = inside.sum() * 4 / N
error = abs((pi - np.pi) / pi) * 100
outside = np.invert(inside)
plt.figure(figsize=(8, 8))
plt.plot(x[inside], y[inside], 'b.')
plt.plot(x[outside], y[outside], 'r.')
plt.plot(0, 0, label=f'π*= {pi:4.3f}error = {error:4.3f}', alpha=0)
plt.axis('square')
plt.xticks([])
plt.yticks([])
plt.legend(loc=1, frameon=True, framealpha=0.9)
plt.show()

N = 10000
x, y = np.random.uniform(-1, 1, size=(2, N))
inside = (x * 2 + y * 2) <= 1
pi = inside.sum() * 4 / N
error = abs((pi - np.pi) / pi) * 100
outside = np.invert(inside)
plt.figure(figsize=(8, 8))
plt.plot(x[inside], y[inside], 'b.')
plt.plot(x[outside], y[outside], 'r.')
plt.plot(0, 0, label=f'π*= {pi:4.3f}error = {error:4.3f}', alpha=0)
plt.axis('square')
plt.xticks([])
plt.yticks([])
plt.legend(loc=1, frameon=True, framealpha=0.9)
plt.show()
