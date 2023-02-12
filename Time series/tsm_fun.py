# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 12:10:56 2023

@author: ilarr
"""
import statsmodels.api as sm
import matplotlib.pyplot as plt

from pmdarima import auto_arima
from arch import arch_model
import importlib
import file_class1
importlib.reload(file_class1)
import pandas as pd
from sklearn.preprocessing import power_transform
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.optimize import minimize
import file_function1
importlib.reload(file_function1)
import numpy as np
import tsm_fun
importlib.reload(tsm_fun)





def find_best_order(data):
    stepwise_fit = auto_arima(data, start_p=1, start_q=1,
                                 max_p=3, max_q=3, m=12,
                                 start_P=0, seasonal=True,
                                 d=None, max_D=2, trace=True,
                                 error_action='ignore',  # we don't want to know if an order does not work
                                 suppress_warnings=True,  # we don't want convergence warnings
                                 stepwise=True)  # set to stepwise

    best_order = stepwise_fit.order
    seasonal_order = stepwise_fit.seasonal_order
    return best_order, seasonal_order

def fit_model(train, order, seasonal_order=None):
    """
    Fits a SARIMAX model to the training data
    
    Parameters:
    - train (Pandas DataFrame): the training data
    - order (tuple): the order of the SARIMAX model, in the format (p,d,q)(P,D,Q,s)
    - seasonal_order (tuple): the seasonal order of the SARIMAX model, in the format (P,D,Q,s)
    - logged (bool): whether to fit the model on the log-transformed data
    
    Returns:
    - SARIMAXResults object: the results of the model fit
    - Pandas DataFrame: the residuals
    """
    
    
    model = sm.tsa.statespace.SARIMAX(train, trend='c', order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    residuals = model_fit.resid
    return model_fit, residuals

def forecast_values(model_fit, test, logged=False):
    """
    Forecasts future values using the model fit
    
    Parameters:
    - model_fit (SARIMAXResults object): the fitted model
    - test (Pandas DataFrame): the test data
    - logged (bool): whether the data is log-transformed
    
    Returns:
    - Pandas DataFrame: the forecasted values with their confidence intervals
    """
    pred_uc = model_fit.get_forecast(len(test))
    pred_ci = pred_uc.conf_int()
    pred_uc.predicted_mean = (pred_uc.predicted_mean)
    pred_ci['predict'] =list((pred_uc.predicted_mean))
    if logged:
        pred_ci['lower close'] = np.exp(pred_ci['lower close'])
        pred_ci['predict'] = np.exp(pred_ci['predict'])
        pred_ci['upper close'] = np.exp(pred_ci['upper close'])
    return pred_ci

def plot_forecast(train, test,model_fit, pred_ci, steps=None):
    """
    Plots the forecasted values with their confidence intervals
    
    Parameters:
    - train (Pandas DataFrame): the training data
    - test (Pandas DataFrame): the test data
    - pred_ci (Pandas DataFrame): the forecasted values with their confidence intervals
    - steps (int): the number of steps to forecast
    - logged (bool): whether the data is log-transformed
    """
    
    pred_uc = model_fit.get_forecast(steps)
    ax = train.plot(label='Data', figsize=(14, 7))
    ax = test.plot(label='Test', figsize=(14, 7))
    pred_ci.predict.plot(ax=ax, label='Predicts')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Days')
    ax.set_ylabel('Price')
    plt.legend()
    plt.show()
    return pred_uc

def compare_forecast(predict, test):
    """
    Plots the forecasted values against the real values
    
    Parameters:
    - pred_ci (Pandas DataFrame): the forecasted values with their confidence intervals
    - test (Pandas DataFrame): the test data
    - logged (bool): whether the data is log-transformed
    """
    
    plt.figure(figsize=(16,9))
    plt.grid(True)
    plt.plot(predict, color = 'blue',marker = 'o',linestyle = 'dashed', label = 'Forecast')
    plt.plot(test, color = 'red', label = 'Precio real')
    plt.show()
    
def plot_residuals(model_fit):
    """
    Plots the residuals of the model fit
    """
    residuals = model_fit.resid[1:]
    fig, ax = plt.subplots(1,2)
    residuals.plot(title='Residuals', ax=ax[0])
    residuals.plot(title='Density', kind='kde', ax=ax[1])
    plt.show()
    
def check_residuals(model_fit):
    """
    Checks the residuals of the model fit
    
    Parameters:
    - model_fit (SARIMAXResults object): the fitted model
    
    Returns:
    - Pandas DataFrame: the Ljung-Box test statistics for the residuals
    """
    return sm.stats.acorr_ljungbox(model_fit.resid, lags=[1],return_df= True)

def print_summary(model_fit):
    """
    Prints a summary of the model fit
    """
    print(model_fit.summary())


