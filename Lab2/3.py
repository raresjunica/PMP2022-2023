import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az
import random

np.random.seed(1)

TailsTails, TailsHead, HeadTails, HeadHead = [], [], [], []
for counter in range(100):
    TailsTailsC, TailsHeadC, HeadTailsC, HeadHeadC = 0, 0, 0, 0
    for throws in range(10):
        firstCoin = np.random.randint(0, 100)
        secondCoin = np.random.randint(0, 100)
        if firstCoin < 50 and secondCoin < 30:
            TailsTailsC += 1
        elif firstCoin < 50 and secondCoin > 30:
            TailsHeadC += 1
        elif firstCoin > 50 and secondCoin < 30:
            HeadTailsC += 1
        elif firstCoin > 50 and secondCoin > 30:
            HeadHeadC += 1
    TailsTails.append(TailsTailsC)
    TailsHead.append(TailsHeadC)
    HeadTails.append(HeadTailsC)
    HeadHead.append(HeadHeadC)
az.plot_posterior({'TAILS TAILS': TailsTails, 'TAILS HEAD': TailsHead, 'HEAD TAILS': HeadTails, 'HEAD HEAD': HeadHead })
plt.show()