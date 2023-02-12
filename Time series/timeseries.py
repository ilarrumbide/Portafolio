# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 10:53:34 2023

@author: ilarr
"""
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
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

class TimeSeriesModel:
    def __init__(self, train, test):
        """
        Initializes the class with training and test data
        
        Parameters:
        - train (Pandas DataFrame): the training data
        - test (Pandas DataFrame): the test data
        """
        self.train = train
        self.test = test
    
    def fit_model(self, order,seasonal_order=None):
        """
        Fits a SARIMAX model to the training data
        
        Parameters:
        - order (tuple): the order of the SARIMAX model, in the format (p,d,q)(P,D,Q,s)
        
        Returns:
        - SARIMAXResults object: the results of the model fit
        """
        model = sm.tsa.statespace.SARIMAX(self.train, trend='c', order=order,seasonal_order=seasonal_order)
        self.model_fit = model.fit()
        residuals = self.model_fit.resid
        return self.model_fit, residuals
        
    def print_summary(self):
        """
        Prints a summary of the model fit
        """
        print(self.model_fit.summary())
        
    def check_residuals(self):
        """
        Checks the residuals of the model fit
        
        Returns:
        - Pandas DataFrame: the Ljung-Box test statistics for the residuals
        """
        return sm.stats.acorr_ljungbox(self.model_fit.resid, lags=[1],return_df= True)
        
    def plot_residuals(self):
        """
        Plots the residuals of the model fit
        """
        residuals = self.model_fit.resid[1:]
        fig, ax = plt.subplots(1,2)
        residuals.plot(title='Residuals', ax=ax[0])
        residuals.plot(title='Density', kind='kde', ax=ax[1])
        plt.show()
        
    def forecast(self):
        """
        Forecasts future values using the model fit
        
        Returns:
        - Pandas DataFrame: the forecasted values with their confidence intervals
        """
        pred_uc = self.model_fit.get_forecast(len(self.test))
        pred_ci = pred_uc.conf_int()
        pred_uc.predicted_mean = (pred_uc.predicted_mean)
        pred_ci['predict'] =list((pred_uc.predicted_mean))
        return pred_ci
        
    def plot_forecast(self,pred_ci,steps= None):
        """
       Plots the forecasted values with their confidence intervals
       
       Parameters:
       - pred_ci (Pandas DataFrame): the forecasted values with their confidence intervals
       """
        
        pred_uc = self.model_fit.get_forecast(steps)
        ax = self.train.plot(label='Data', figsize=(14, 7))
        ax = self.test.plot(label='Test', figsize=(14, 7))
        #ax = pred_uc.plot(label = 'future',figsize = (14, 7))
        pred_ci.predict.plot(ax=ax, label='Predicts')
        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.25)
        ax.set_xlabel('Days')
        ax.set_ylabel('Price')
        plt.legend()
        plt.show()
        return pred_uc
        
    def compare_forecast(self,pred_ci):
        plt.figure(figsize=(16,9))
        plt.grid(True)
        plt.plot(pred_ci, color = 'blue',marker = 'o',linestyle = 'dashed', label = 'Forecast')
        plt.plot(self.test, color = 'red', label = 'Precio real')
 
    def find_best_order(self):
        """
        Finds the best order for the SARIMAX model
        
        Returns:
        - tuple: the best order for the SARIMAX model, in the format (p,d,q)(P,D,Q,s)
        """
        stepwise_model = auto_arima(self.train, start_p=1, start_q=1,
                                max_p=3, max_q=3, m=20,
                                start_P=0, seasonal=True,
                                d=1, max_D=2, trace=True,
                                error_action='ignore',  
                                suppress_warnings=True, 
                                stepwise=True)
        params = stepwise_model.get_params()
        return params['order'], params['seasonal_order']
    
    

    def modify_forecast(self):
        # Get the predictions from the SARIMAX model
        pred_uc = self.model_fit.get_forecast(len(self.test))
        pred_mean = pred_uc.predicted_mean

        # Calculate the residuals of the SARIMAX model
        resids = self.test - pred_mean
        # Fit a GARCH model to the residuals of the SARIMAX model
        garch_model = arch_model(resids, mean='Constant', vol='GARCH', p=1, q=1,dist = 'skewt')
        garch_fit = garch_model.fit(disp='off')
        self.garch_fit = garch_fit

        # Use the GARCH model to modify the predictions
        modified_preds = pred_mean + garch_fit.resid
        
        print(garch_fit.resid)

        return modified_preds, garch_fit.resid
    
    def forecast_future(self, steps):
       """
       Forecasts future values beyond the test data
       
       Parameters:
       - steps (int): the number of steps to forecast
       
       Returns:
       - forecast (SARIMAXForecast object): the forecasted mean values and confidence intervals
       """
       model_full = sm.tsa.statespace.SARIMAX(pd.concat([self.train, self.test]), trend='c', order=order)
       self.model_fit_full = model_full.fit()
       residuals = self.model_fit_full.resid
       forecast = self.model_fit_full.get_forecast(steps=steps)
       forecast_mean = forecast.predicted_mean
       forecast_conf_int = forecast.conf_int()
       return forecast_mean, forecast_conf_int, residuals
   
    def plot_future_forecast(self, forecast_mean, forecast_conf_int):
        """
        Plots the forecasted future values with their confidence intervals
        
        Parameters:
        - forecast_mean (Pandas Series): the forecasted mean values
        - forecast_conf_int (Pandas DataFrame): the forecasted confidence intervals
        """
        ax = self.train.plot(label='Data', figsize=(14, 7))
        ax = self.test.plot(label='Test', figsize=(14, 7))
        forecast_mean.plot(ax=ax, label='Future Forecast')
        ax.fill_between(forecast_conf_int.index,
                        forecast_conf_int.iloc[:, 0],
                        forecast_conf_int.iloc[:, 1], color='k', alpha=.25)
        ax.set_xlabel('Days')
        ax.set_ylabel('Price')
        plt.legend()
        plt.show()

    def add_garch(self, resids,step,a):
        """
        Fits a GARCH model to the residuals of the SARIMAX model
        
        Parameters:
        - resids (Pandas DataFrame): the residuals of the SARIMAX model
        
        Returns:
        - numpy array: the predictions from the GARCH model
        """
        resids = resids / a
        garch_model = arch_model(resids, mean='Constant', vol='GARCH', p=1, q=1, dist='skewt')
        garch_fit = garch_model.fit(disp='off')
        self.garch_fit = garch_fit
        garch_predictions = garch_fit.forecast(horizon=step)
        garch_fit.resid
        return garch_predictions.residual_variance.values[-1], garch_fit.resid

    
   


import numpy as np

inputs = file_class1.distribution_input()
inputs.data_type = 'real' # real simulation
inputs.variable_name = 'BTC-USD' # TXR.BA , ALUA.BA , ^MERV
inputs.degrees_freedom = None
inputs.nb_sims = None

tm = file_class1.time_series(inputs)
t = tm.load_timeseries()

L = file_function1.split_ts(t,0.98,0.02)
train_data, test_data = file_function1.get_train_test(L)
tsm = TimeSeriesModel((train_data.close), (test_data.close))
best_order, seasonal_order = tsm.find_best_order()
order = best_order
results, resid = tsm.fit_model((0,1,0)(1,1,1))
residual, raro = tsm.add_garch(resid,step=8,a=1000)
tsm.print_summary()
tsm.check_residuals()
tsm.plot_residuals()
forecast = (tsm.forecast())
tsm.plot_forecast(forecast,100)
tsm.compare_forecast(forecast.predict)
modify_forecast, r1 = tsm.modify_forecast()
tsm.compare_forecast(modify_forecast)
forecast_mean, forecast_conf_int, residuals = tsm.forecast_future(steps = 20)
tsm.plot_future_forecast(forecast_mean, forecast_conf_int)



mse = mean_squared_error(test_data.close, forecast.predict)
mae = mean_absolute_error(test_data.close, forecast.predict)
print("MSE: ", mse)
print("MAE: ", mae)

