import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az
import math
from scipy import stats

if __name__ == '__main__':
    data = pd.read_csv("Prices.csv")
    Price = data['Price'].values
    Speed = data['Speed'].values
    HardDrive = data['HardDrive'].values
    Ram = data['Ram'].values
    Premium = data['Premium'].values
    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sd=1)
        beta1 = pm.Normal('beta1', mu=0, sd=1)
        beta2 = pm.Normal('beta2', mu=0, sd=1)
        sigma = pm.HalfNormal('sigma', sd=1)
        mu = pm.Deterministic('mu', alpha + beta1 * Speed + beta2 * [math.log(i) for i in HardDrive])
        Price_obs = pm.Normal('price_obs', mu=mu, sigma=sigma, observed=Price)
        trace = pm.sample(5000, tune=1000)
        predictive = pm.sample_posterior_predictive(trace, var_names=["mu"], samples=5000)
    az.plot_posterior(
        {"beta1": trace['beta1'], "beta2": trace['beta2']},
        hdi_prob=0.95)
    plt.show()
    az.plot_posterior(
        {"pret asteptat": trace['mu']},
        hdi_prob=0.9)
    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sd=1)
        beta = pm.Normal('beta', mu=0, sd=1)
        sigma = pm.HalfNormal('sigma', sd=1)
        mu = pm.Deterministic('mu', alpha + beta * [0 if i == "no" else 1 for i in Premium])
        Price_obs = pm.Normal('price_obs', mu=mu, sigma=sigma, observed=Price)
        trace = pm.sample(5000, tune=1000)
    az.plot_posterior(
        {"premium_beta": trace['beta']})
    plt.show()

# Media beta ~0 => producatorul premium are impact prea mare

# ex3
# Avand in vedere deviatia standard, cat si faptul ca beta1 si beta2 sunt diferite de 0 putem spune ca sunt
# indicatori utili ai pretului
