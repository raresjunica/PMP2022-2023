import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az

np.random.seed(1)
if __name__ == '__main__':
    data = pd.read_csv('Prices.csv')

    price = data['Price'].values
    speed = data['Speed'].values
    hardDrive = data['HardDrive'].values
    ram = data['Ram'].values
    premium = data['Premium'].values

    # fig, axes = plt.subplots(2, 2, sharex=False, figsize=(10, 8))
    # axes[0, 0].scatter(speed, price, alpha=0.6)
    # axes[0, 1].scatter(hardDrive, price, alpha=0.6)
    # axes[1, 0].scatter(ram, price, alpha=0.6)
    # axes[1, 1].scatter(premium, price, alpha=0.6)
    # axes[0, 0].set_ylabel("Price")
    # axes[0, 0].set_xlabel("Speed")
    # axes[0, 1].set_xlabel("HardDrive")
    # axes[1, 0].set_xlabel("Ram")
    # axes[1, 1].set_xlabel("Premium")
    # plt.savefig('price_correlations.png')

    x = np.array(data["Speed"], np.log(data["HardDrive"]))
    x_mean = x.mean(axis=0, keepdims=True)
    x_centered = x - x_mean
    with pm.Model() as model:
        a_tmp = pm.Normal('a_tmp', mu=0, sd=10)
        beta_1 = pm.Normal('beta_1', mu=0, sigma=10)
        beta_2 = pm.Normal('beta_2', mu=0, sigma=10)
        epsilon = pm.HalfCauchy('epsilon', 5)
        miu = a_tmp + speed*beta_1 + hardDrive*beta_2
        a = pm.Deterministic('a', a_tmp - pm.math.dot(x_mean, beta_1))
        y_pre = pm.Normal('y_pre', mu=miu, sd=epsilon, observed=price)
        step = pm.Slice()
        step = pm.Metropolis()
        idata_model = pm.sample(2000, step=step, tune=2000, return_inferencedata=True)
