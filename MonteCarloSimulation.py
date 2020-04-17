import numpy as np
import pandas as pd
import pandas_datareader as dr
import matplotlib.pyplot as plt
from scipy.stats import norm

# Fetch data
data = pd.DataFrame()
data['AAPL'] = dr.DataReader(
    'AAPL', data_source='yahoo', start='2019-01-01')['Adj Close']

# Transforming data
log_returns = np.log(1+data.pct_change())
df = pd.concat([data, log_returns], axis=1)
ax = df.plot(subplots=True)
ax[0].set(ylabel='returns')
ax[1].set(ylabel='log_returns')
plt.show()

# Setting up drift and random components
mean = log_returns.mean()
var = log_returns.var()
std = log_returns.std()
drift = mean - (0.5*var)

intervals = 30  # 1 month of simulation
iterations = 30  # Number of simulations
returns = np.exp(drift.values + std.values *
                 norm.ppf(np.random.rand(intervals, iterations)))

# Set the latest close price at the starting point for the simulation
start = data.iloc[-1]
price_list = np.zeros_like(returns)  # Create a array of 0 with the same shape
price_list[0] = start

for i in range(1, intervals):
    price_list[i] = price_list[i-1]*returns[i]

plt.plot(price_list)
plt.show()
