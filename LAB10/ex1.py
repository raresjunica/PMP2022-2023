import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import pymc3 as pm

#if __name__ == '__main__':

#1
clusters = 3
n_cluster = [100, 200, 200]
n_total = sum(n_cluster)
means = [5, 0, 2]
std_devs = [2, 2, 2]
mix = np.random.normal(np.repeat(means, n_cluster),
np.repeat(std_devs, n_cluster))
az.plot_kde(np.array(mix));
plt.show()