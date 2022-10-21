import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az
from numpy import random

model = pm.Model()
x = np.random.poisson(20, 1)
alpha = 14

with model:
    y = pm.Normal('Timp', sigma=0.5, mu=1)
    z = pm.Exponential('Comanda', 1 / alpha)
    trace = pm.sample(x[0], chains=1)

dictionary = {
    'timp': trace['Timp'].tolist(),
    'comanda': trace['Comanda'].tolist()
}
df = pd.DataFrame(dictionary)

az.plot_posterior(trace)
plt.show()
