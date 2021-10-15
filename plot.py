import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("./results/metrics/simple_tag_2Run_DDQN_seed2_2x64.csv", delimiter=",")
plt.axis([0,5000,-4,4])
#print(data.head(100))
plt.plot(data["Epoch"].head(5000), data["CM_Reward"].head(5000),data["Epoch"].head(5000), data["CM_Adv_Reward"].head(5000))
#plt.plot(data["Epoch"].head(10),data["Ep_Adv_Reward"].head(10))
plt.show()