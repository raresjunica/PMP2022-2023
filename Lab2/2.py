import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az
import random

np.random.seed(1)

s1 = stats.gamma.rvs(4, 0, 1/3)
s2 = stats.gamma.rvs(4, 0, 1/2)
s3 = stats.gamma.rvs(5, 0, 1/2)
s4 = stats.gamma.rvs(5, 0, 1/3)

print('Server 1:', s1)
print('Server 2:', s2)
print('Server 3:', s3)
print('Server 4:', s4)

x = []
for i in range(100):
    y = np.random.randint(0, 100)
    if y < 25:
        x.append(stats.gamma.rvs(4, 0, 0.33, 1)[0] + stats.expon.rvs(0, 0.25, 1)[0])
    elif y < 50:
        x.append(stats.gamma.rvs(4, 0, 0.5, 1)[0] + stats.expon.rvs(0, 0.25, 1)[0])
    elif y < 80:
        x.append(stats.gamma.rvs(5, 0, 0.5, 1)[0] + stats.expon.rvs(0, 0.25, 1)[0])
    else:
        x.append(stats.gamma.rvs(5, 0, 0.33, 1)[0] + stats.expon.rvs(0, 0.25, 1)[0])

i = 0
for ms in x:
    if ms > 3:
        i = i + 1

i = i / 100
print(i)
az.plot_posterior({'x': x})
plt.show()