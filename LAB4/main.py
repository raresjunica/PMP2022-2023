from numpy import random

import arviz as az
import matplotlib.pyplot as plt

import numpy as np
import pymc3 as pm
import pandas as pd

df = pd.read_csv('data.csv')

plt.scatter(df.ppvt, df.momage, color="b", s=100)
plt.xticks(rotation=25)
plt.xlabel('PPVT')
plt.ylabel('Mom Aage')
plt.title('PPVT Report', size=20)

plt.show()

average_age = 0
average_ppvt = 0
average_educ_cat = 0
for age in df.momage:
    average_age = average_age + age
for ppvt in df.ppvt:
    average_ppvt = average_ppvt + ppvt
for educ in df.educ_cat:
    average_educ_cat = average_educ_cat + educ


average_age = average_age / df.index.stop
average_ppvt = average_ppvt / df.index.stop
average_educ_cat = average_educ_cat / df.index.stop


with pm.Model() as model_g:
    mom_age = pm.Normal('mom_age', mu=average_age, sd=1)
    ppvt = pm.Normal('ppvt', mu=average_ppvt, sd=1)
    educ_cat = pm.Normal('educ_cat', mu=average_educ_cat, sd=1)

map_estimate = pm.find_MAP(model=model_g)
print(map_estimate)

_, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].plot(mom_age, ppvt, 'CO.')
ax[0].set_xlabel('Mom Age')
ax[0].set_ylabel('PPVT')
plt.tight_layout()
plt.show()

