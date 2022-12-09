import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az


# EXERCITIUL 1
az.style.use('arviz-darkgrid')
dummy_data = np.loadtxt('date.csv')
x_1 = dummy_data[:, 0]
y_1 = dummy_data[:, 1]
order = 5
x_1p = np.vstack([x_1 ** i for i in range(1, order + 1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean()) / y_1.std()
plt.scatter(x_1s[0], y_1s)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

with pm.Model() as model_p2:
    alpha_1 = pm.Normal('α', mu=0, sd=1)
    beta_1 = pm.Normal('β', mu=0, sd=10, shape=order)
    eps_1 = pm.HalfNormal('ε', 5)
    miu_1 = alpha_1 + pm.math.dot(beta_1, x_1s)
    y_pred_2 = pm.Normal('y_pred', mu=miu_1, sd=eps_1, observed=y_1s)
    idata_p_1 = pm.sample(2000, return_inferencedata=True)

x_new_2 = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)
α_p_post_1 = idata_p_1.posterior['α'].mean(("chain", "draw")).values
β_p_post_1 = idata_p_1.posterior['β'].mean(("chain", "draw")).values
idx = np.argsort(x_1s[0])
y_p_post_1 = α_p_post_1 + np.dot(β_p_post_1, x_1s)
plt.plot(x_1s[0][idx], y_p_post_1[idx], 'C2', label=f'model order {order}')
plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
plt.legend()
plt.show()

with pm.Model() as model_p:
    alpha = pm.Normal('α', mu=0, sd=1)
    beta = pm.Normal('β', mu=0, sd=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=order)
    eps = pm.HalfNormal('ε', 5)
    miu = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal('y_pred', mu=miu, sd=eps, observed=y_1s)
    idata_p = pm.sample(2000, return_inferencedata=True)

x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)
α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
idx = np.argsort(x_1s[0])
y_p_post = α_p_post + np.dot(β_p_post, x_1s)
plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order}')
plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
plt.legend()

with pm.Model() as model_p2:
    alpha_2 = pm.Normal('α', mu=0, sd=1)
    beta_2 = pm.Normal('β', mu=0, sd=100, shape=order)
    eps_2 = pm.HalfNormal('ε', 5)
    miu_2 = alpha_2 + pm.math.dot(beta_2, x_1s)
    y_pred_2 = pm.Normal('y_pred', mu=miu_2, sd=eps_2, observed=y_1s)
    idata_p_2 = pm.sample(2000, return_inferencedata=True)

x_new_2 = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)
α_p_post_2 = idata_p.posterior['α'].mean(("chain", "draw")).values
β_p_post_2 = idata_p.posterior['β'].mean(("chain", "draw")).values
idx = np.argsort(x_1s[0])
y_p_post_2 = α_p_post_2 + np.dot(β_p_post, x_1s)
plt.plot(x_1s[0][idx], y_p_post_2[idx], 'C2', label=f'model order {order}')
plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
plt.legend()


#EXERCITIUL 2
x_2 = np.random.uniform(low=min(x_1), high=max(x_1), size=(500,))
y_2 = np.random.uniform(low=min(y_1), high=max(y_1), size=(500,))
x_2p = np.vstack([x_2 ** i for i in range(1, order + 1)])
x_2s = (x_2p - x_2p.mean(axis=1, keepdims=True)) / x_2p.std(axis=1, keepdims=True)
x_2p.std(axis=1, keepdims=True)
y_2s = (y_2 - y_2.mean()) / y_2.std()
plt.scatter(x_2s[0], y_2s)

#EXERCITIUL 3
order = 2
idata_l = pm.sample(2000, return_inferencedata=True)
waic_l = az.waic(idata_l, scale="deviance")
loo_l = az.loo(idata_l, scale="deviance")
waic_l = az.waic(idata_l, scale="deviance")
plt.show()