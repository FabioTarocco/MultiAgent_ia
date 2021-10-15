import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("./results/metrics/simple_tag_DDQN_seed1_2x64_gamma_10.csv", delimiter=",")
plt.axis([0,5000,-40,40])
#print(data.head(100))
plt.plot(data["Epoch"].head(5000), data["Ep_Reward"].head(5000),data["Epoch"].head(5000), data["Ep_Adv_Reward"].head(5000))
#plt.plot(data["Epoch"].head(10),data["Ep_Adv_Reward"].head(10))
plt.show()