import pandas as pd
import pandas_datareader as dr
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm


# Fetch data
tickers = ['AAPL', 'AMZN', 'FB']
data = dr.DataReader(tickers, data_source='yahoo',
                     start='2018-01-01')['Adj Close']
returns = data.pct_change()
cov_matrix = returns.cov()
avg_returns = returns.mean()

# Set portfolio weights and investment
weights = np.array([0.25, 0.3, 0.45])
investment = 1000000

# Calculating portfolio stats
port_avg_returns = avg_returns.dot(weights)
port_std = np.sqrt(weights.transpose().dot(cov_matrix).dot(weights))
invest_mean = (1+port_avg_returns)*investment
invest_std = port_std*investment

# Historical VAR 1-day
ppf = norm.ppf(0.05, invest_mean, invest_std)
var1d = investment - ppf
print(var1d)

# Historical VAR n-day
var_array = []
num_days = int(10)
for i in range(1, num_days+1):
    var = np.round(var1d*np.sqrt(i), 4)
    var_array.append(var)
    print(str(i) + '-day VAR at 95% confidence:{}'.format(var))

# Plot maximum lost over n-day periods
plt.xlabel('Day')
plt.ylabel('Max portfolio loss(USD)')
plt.title('Max portfolio VAR over {}-day period'.format(num_days))
plt.plot(var_array)
plt.show()

# Compare returns vs normal distribution
stock = ['AMZN']
returns[stock].hist(density=True, histtype='stepfilled')
x = np.linspace(port_avg_returns-3*port_std,
                port_avg_returns+3*port_std)
plt.plot(x, norm.pdf(x, port_avg_returns, port_std))
plt.show()
