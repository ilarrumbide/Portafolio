# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 13:58:29 2023

@author: ilarr
"""
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from arch import arch_model
import importlib

import pandas as pd
from sklearn.preprocessing import power_transform
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.optimize import minimize
import numpy as np
import yfinance as yf
import file_function1
importlib.reload(file_function1)

import tsm_fun
importlib.reload(tsm_fun)

import file_class1
importlib.reload(file_class1)

inputs = file_class1.distribution_input()
inputs.data_type = 'real' # real simulation
inputs.variable_name = 'ALUA.BA' # TXR.BA , ALUA.BA , ^MERV
inputs.degrees_freedom = None
inputs.nb_sims = None

dm = file_class1.distribution_manager(inputs) # initialise constructor
dm.load_timeseries() # get the timeseries
dm.compute() # compute returns and all different risk metrics
dm.plot_histogram() # plot histogram
print(dm) # write all data in console
dm.percentile(80)
    

tm = file_class1.time_series(inputs)
t = tm.load_timeseries()


tm.plot_timeseries()

tm.descomposicion()
tm.test_stationarity()
tm.check_stationarity(t.close.diff().dropna())
tm.descomposicion(t.close.diff().diff().dropna())
tm.correlation()


L = file_function1.split_ts(t,0.95,0.05)
train_data, test_data = file_function1.get_train_test(L)

train = np.log(train_data.close)
test = np.log(test_data.close)

order,seasonal_order = tsm_fun.find_best_order(train)
model_fit, residuals = tsm_fun.fit_model(train,order,seasonal_order)
model_fit.plot_diagnostics(figsize = (14,9))

forecast = model_fit.get_forecast(steps=len(test))
pred_ci = tsm_fun.forecast_values(model_fit,test,logged= True)    

tsm_fun.plot_forecast(train_data.close, test_data.close, model_fit, (pred_ci))
tsm_fun.compare_forecast(pred_ci.predict, test_data.close)

mse = mean_squared_error(test_data.close, pred_ci.predict)
mae = mean_absolute_error(test_data.close, pred_ci.predict)
print("MSE: ", mse)
print("MAE: ", mae)


model_fit_full, residuals_full = tsm_fun.fit_model(t.close,order,seasonal_order)
forecast_future = model_fit_full.get_forecast(steps=10)
pred_ci_full = forecast_future.conf_int()
pred_ci_full['predict'] = forecast_future.predicted_mean 

future = pd.DataFrame()
future = pd.concat([train_data.close,test_data.close,pred_ci_full.predict,pred_ci.predict],axis = 1, join = 'outer')

ax = train_data.close.plot(label='Data', figsize=(14, 7))
ax = test_data.close.plot(label='Test', figsize=(14, 7))
pred_ci_full.predict.plot(ax=ax, label='Predicts')
pred_ci.predict.plot(ax=ax, label='Predicts_test')
ax.set_xlabel('Days')
ax.set_ylabel('Price')
plt.legend()
plt.show()




from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import plotly.offline as py
py.init_notebook_mode()


my_model = Prophet(interval_width=0.95)
my_model.fit(df)

future_dates = my_model.make_future_dataframe(periods=36, freq='MS')
future_dates.head()

forecast = my_model.predict(future_dates)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()

my_model.plot(forecast, uncertainty=True)

my_model.plot_components(forecast)

from fbprophet.plot import add_changepoints_to_plot
fig = my_model.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), my_model, forecast)

pro_change= Prophet(n_changepoints=20, yearly_seasonality=True, changepoint_prior_scale=0.08)
forecast = pro_change.fit(df).predict(future_dates)
fig= pro_change.plot(forecast);
a = add_changepoints_to_plot(fig.gca(), pro_change, forecast)


import pandas as pd
import datetime as dt
start = dt.datetime(2011, 1, 1)
end = dt.datetime(2021, 1, 1)
stock_data = yf.download('ALUA.BA',period = '3y')
returns = stock_data['Adj Close'][0:717].pct_change().dropna()
test = stock_data['Adj Close'][717:]
daily_vol = returns.std()
vol = daily_vol * np.sqrt(252)



T = len(test)
T = 16
count = 0
price_list = []
last_price = stock_data['Adj Close'][0:717][-1]





# Estimate GARCH model
garch_model = arch_model(returns.dropna(), mean='constant', vol='GARCH', p=1, q=1, dist='skewt')
results = garch_model.fit()
volatility = results.conditional_volatility




def monte_carlo_simulation(last_price, volatility, T, NUM_SIMULATIONS):
    df = pd.DataFrame()
    last_price_list = []
    for x in range(NUM_SIMULATIONS):
        count = 0
        price_list = []
        price = last_price
        price_list.append(price)

        for y in range(T):
            if count == 16:
                break
            price = price_list[count]* (1 + np.random.normal(0, volatility[count]))
            price_list.append(price)
            count += 1

        df[x] = price_list
        last_price_list.append(price_list[-1])
    mean_predictions = df.mean(axis=1)
    prediction_df = pd.DataFrame(mean_predictions, columns=["prediction"])
    return df,prediction_df, last_price_list

NUM_SIMULATIONS =  1000
df,prediction_df,last_price_list = monte_carlo_simulation(last_price, volatility, T, NUM_SIMULATIONS)


        
fig = plt.figure()
fig.suptitle("Monte Carlo Simulation: MSFT")
plt.plot(df)
plt.xlabel('Day')
plt.ylabel('Price')
plt.show()

print("Expected price: ", round(np.mean(last_price_list),2))
print("Quantile (5%): ",np.percentile(last_price_list,5))
print("Quantile (95%): ",np.percentile(last_price_list,95))

plt.hist(last_price_list,bins=100)
plt.axvline(np.percentile(last_price_list,5), color='r', linestyle='dashed', linewidth=2)
plt.axvline(np.percentile(last_price_list,95), color='r', linestyle='dashed', linewidth=2)
plt.show()

p = pd.DataFrame()
p['price'] = test_data.close
p['sim'] = price_list[1:]
p['predict'] = pred_ci.predict

NUM_SIMULATIONS =  1000
df = pd.DataFrame()
last_price_list = []
for x in range(NUM_SIMULATIONS):
    count = 0
    price_list = []
    price = last_price * (1 + np.random.normal(0, daily_vol))
    price_list.append(price)
    
    for y in range(T):
        if count == 251:
            break
        price = price_list[count]* (1 + np.random.normal(0, daily_vol))
        price_list.append(price)
        count += 1
        
    df[x] = price_list
    last_price_list.append(price_list[-1])

new = pd.DataFrame(columns=["simulation"])

# Add the mean of all the simulations to the new DataFrame
new["simulation"] = df.mean(axis = 1)





close = stock_data.Close
train = close[0:717]
test = close[717:]

import matplotlib.pyplot as plt

# calculate Hurst of recent prices
import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from numpy.random import randn
from numpy import random as rn


S0 = train[-1]
mu = returns.mean()
sigma = np.std(returns)
M = 1000
N = 100
T = len(test)
h = T/N
Z = rn.randn(M,N)
S = S0*np.ones((M,N+1))
for i in range(0,N):
 S[:,i+1] = S[:,i] + S[:,i]*( mu*h + sigma*np.sqrt(h)*Z[:,i] )
plt.figure(figsize=(17,10))
a = [ rn.randint(0,M) for j in range(1,20)]
for runer in a:
 plt.plot(np.arange(0,T+h,h),S[runer])
