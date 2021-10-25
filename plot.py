import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("./results/metrics/simple_spread_DDQN_seed2_2x64_gamma_50.csv", delimiter=",")
plt.axis([0,5000,-30,-20])
#print(data.head(100))
plt.plot(data["Epoch"].head(5000), data["CM_Reward"].head(5000),data["Epoch"].head(5000), data["CM_Adv_Reward"].head(5000))
#plt.plot(data["Epoch"].head(10),data["Ep_Adv_Reward"].head(10))
plt.show()