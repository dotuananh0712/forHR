import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import date
from quantopian.pipeline import Pipeline
# Financial metrics data of equities
from quantopian.pipeline.data import Fundamentals
# Daily US equities pricing
from quantopian.pipeline.data.builtin import USEquityPricing as USEP
# window_length from CustomFactor
from quantopian.pipeline.factors import CustomFactor, Returns
# Filters for stocks 'https://www.quantopian.com/docs/api-reference/pipeline-api-reference#quantopian.pipeline.filters.QTradableStocksUS'
from quantopian.pipeline.filters import QTradableStocksUS as QTSU
from quantopian.research import run_pipeline

# Set date range
start = '2010-01-01'
end = date.today()

# Get returns of equities in date range


def get_returns(start_date, end_date):
    pipeline = Pipeline(
        columns={'Close': USEP.close.latest},
        screen=QTSU()
    )
    stocks = run_pipeline(pipeline, start, end)
    unstacked = stocks.unstack()
    prices = unstacked['Close'].fillna(method='ffill').fillna(
        method='bfill').dropna(axis=1, how='any')
    perc_returns = prices.pct_change()
    return perc_returns


R = get_returns(start, end)
assets = R.columns()

# Calculating Fama-French factors: market size, company size, BP ratio


def make_pipeline():
    market_cap = Fundamentals.shares_outstanding.latest/USEP.close.latest
    book_to_price = 1/Fundamentals.pb_ratio.latest
    # Building a filter universe
    biggest = market_cap.top(500, mask=QTSU)
    smallest = market_cap.bottom(500, mask=QTSU)
    highbp = book_to_price.top(500, mask=QTSU)
    lowbp = book_to_price.bottom(500, mask=QTSU)
    screen = biggest | smallest | highbp | lowbp
    pipeline = Pipeline(
        columns={'returns': Returns, 'market_cap': market_cap, 'book-to-price_ratio': book_to_price,
                 'biggest_cap': biggest, 'smallest_cap': smallest, 'highbp': highbp, 'lowbp': lowbp},
        screen=screen
    )
    return pipeline


pipeline = make_pipeline()
results = run_pipeline(pipeline, start, end)
df_biggest = results[results.biggest]['returns'].groupby(level=0).mean()
df_smallest = results[results.smallest]['returns'].groupby(level=0).mean()
df_highbp = results[results.highbp]['returns'].groupby(level=0).mean()
df_lowbp = results[results.lowbp]['returns'].groupby(level=0).mean()

SMB = df_smallest - df_biggest
HML = df_highbp - df_lowbp

df = pd.DataFrame({
    'SMB': SMB  # company size
    'HML': HML  # company PB ratio
}, columns=['SMB', 'HML']).dropna()

market = get_pricing('SPY', start_date=start, end_date=end,
                     fields='price').pct_change()  # USEP function
market = pd.DataFrame({'market': market})
F = pd.concat([MKT, df], axis=1).dropna()


# Calculating factors risk B
B = pd.DataFrame(index=assets, dtype=np.float)
epsilon = pd.DataFrame(index=R.index, dtype=np.float)

x = sm.addconstant(F)  # adding intercept b to linear else b = 0

for i in assets:  # Removing outliers > 3*std from mean
    y = R.loc[:, i]
    y_inlier = y[np.abs(y-y.mean()) <= (3*y.std())]
    x_inlier = x[np.abs(x-x.mean()) <= (3*x.std())]
    result = sm.OLS(y_inlier, x_inlier).fit()

    B.loc[i, 'MKT_beta'] = result.params[1]
    B.loc[i, 'SMB_beta'] = result.params[2]
    B.loc[i, 'HML_beta'] = result.params[3]
    epsilon.loc[:, i] = y - (x.iloc[:, 0]*result.param[0] + x.iloc[:, 1]*result.params[1] +
                             x.iloc[:, 2]*result.params[2] + x.iloc[:, 3]*result.params[3])  # Fama-French factors formula

# Calculating portfolio risk std = sqrt(common factor variance + specific variance) (Schweser)
# Creating equally weighted by dividing 1 to the total number of assets
w = np.ones([1, R.shape[1]])/R.shape[1]
# Calculating common factor variance: wBVB(transpose)w(transpose)


def common_factor_variance(factors, factor_exposure, w):
    B = np.asarray(factor_exposure)
    F = np.asarray(factors)
    V = np.asarray(factors.cov())

    return w.dot(B.dot(V).dot(B.transpose())).dot(w.transpose())

# Calculating specific variance: wDw(transpose)


def specific_variance(epsilon, w):
    D = np.diag(np.asarray(epsilon.var())) * \
        epsilon.shape[0] / (epsilon.shape[0]-1)

    return w.dot(D).dot(w.transpose())
