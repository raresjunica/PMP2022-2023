import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az

np.random.seed(1)

if __name__ == '__main__':
    model = pm.Model()

    with model:
        # probabilitate cutremur
        cutremur = pm.Bernoulli('C', 0.0005)
        # probabilitate incendiu in functie de cutremur
        incendiu_cutremur = pm.Deterministic('I_c', pm.math.switch(cutremur, 0.03, 0.01))
        incendiu = pm.Bernoulli('I', p=incendiu_cutremur)
        # probabilitate declansare alarma incendiu in functie de cutremur
        alarma_cutremur = pm.Deterministic('A_c', pm.math.switch(cutremur, 0.02, 0.0001))
        alarma = pm.Bernoulli('A', p=alarma_cutremur)
        # probabilitate declansare alarma de incendiu in caz de incendiu
        alarma_i = pm.Deterministic('A_i', pm.math.switch(cutremur, 0.98, 0.95))
        alarma_incendiu = pm.Bernoulli('A_incendiu', p=alarma_i, observed=0)
        trace = pm.sample(20000)

        dictionary = {
            'cutremur': trace['C'].tolist(),
            'incendiu': trace['I'].tolist(),
            'alarma': trace['A'].tolist(),
        }

        dataframe = pd.DataFrame(dictionary)

        p_cutremur = dataframe[(dataframe['incendiu'] == 1)].shape[0] / dataframe.shape[0]

        print(p_cutremur)

        az.plot_posterior(trace)
        plt.show()
