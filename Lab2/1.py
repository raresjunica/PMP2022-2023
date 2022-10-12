import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az
import random

np.random.seed(1)

m1 = stats.expon.rvs(0, 1/4, size=10000)
m2 = stats.expon.rvs(0, 1/6, size=10000)

x = []

for i in range(10000):
    y = random.randint(0, 100)
    if y < 40:
        x.append(stats.expon.rvs(0, 1/4, 1)[0])
    else:
        x.append(stats.expon.rvs(0, 1/6, 1)[0])

az.plot_posterior({'M1': m1, 'M2': m2, 'X': x})
plt.show()



