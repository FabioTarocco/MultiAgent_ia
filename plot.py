import matplotlib.pyplot as plt
import numpy as np
import csv

data = np.genfromtxt("./results/metrics/simple_DDQN_seed1_2x64.csv", delimiter=",", names=["Epochs", "Rewards"])
plt.plot(data['Epochs'], data['Rewards'])
plt.show()