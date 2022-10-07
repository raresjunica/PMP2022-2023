from random import random

import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az
import random

m1 = stats.expon.rvs(0, 1/4, size=10000)
m2 = stats.expon.rvs(0, 1/6, size=10000)

print(m1)
print(m2)

x = []

for i in range(10000):
    y = random.randint(0, 10)
    if y < 4:
        x.append(stats.expon.rvs(0, 1/4, 1)[0])
    else:
        x.append(stats.expon.rvs(0, 1/6, 1)[0])

az.plot_posterior({'M1': m1, 'M2': m2, 'X': x})
plt.show()



