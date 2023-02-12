# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 19:38:44 2023

@author: ilarr
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2, linregress
from numpy import linalg as LA
import yfinance as yf
import seaborn as sns
from pylab import rcParams

from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from arch import arch_model

from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
import statsmodels.tsa.api as smt

import file_class1
importlib.reload(file_class1)
import file_function1
importlib.reload(file_function1)
from scipy.optimize import minimize
       
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
    
    def forecast_future(self, steps,order,season=None):
       """
       Forecasts future values beyond the test data
       
       Parameters:
       - steps (int): the number of steps to forecast
       
       Returns:
       - forecast (SARIMAXForecast object): the forecasted mean values and confidence intervals
       """
       
       model_full = sm.tsa.statespace.SARIMAX(pd.concat([self.train, self.test]), trend='c',  order=order,season=season)
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
    
class time_series():
    
    def __init__(self, inputs):
        self.inputs = inputs # distribution_inputs
        self.description = None
        self.close = None
        self.initialize()
        
    def initialize(self):
        t = self.load_timeseries()
        ric = self.inputs.variable_name
        self.vec_close = t['close']
        self.nb_rows = t.shape[0]
        self.description = "Time Series" +  ric
        
    
    def __str__(self):
        str_self = self.description + ' | size ' + str(self.nb_rows) + '\n' + self.plot_str()
        return str_self
    
    def load_timeseries(self):
        ric = self.inputs.variable_name
        t = file_function1.load_timeseries(ric)      
              
        return t
     
     
    
    def plot_timeseries(self):
        
        self.vec_close.plot(figsize=(12,6), fontsize=14)        
        plt.title(self.description)
        plt.xlabel("Days")
        plt.show()
        
    def plot_timeseries1(self, t = None):
        
        if t is None:
            self.vec_close.plot(figsize=(12,6), fontsize=14) 
        else:
            t.plot(figsize=(12,6), fontsize=14)        
        plt.title(self.description)
        plt.xlabel("Days")
        plt.show()
    
   
        
    def descomposicion(self,t= None):
        rcParams['figure.figsize'] = 14, 8
        if t is None:
            decomposition = sm.tsa.seasonal_decompose(self.vec_close, model='additive',period=20 )
        else:
            decomposition = sm.tsa.seasonal_decompose(t, model='additive',period=20 )
        fig = decomposition.plot()
        plt.show()
        
    def test_stationarity(self):
        
        ventana_de_tiempo = 20
        #Determing rolling statistics
        rolmean = pd.Series(self.vec_close).rolling(window=ventana_de_tiempo).mean()
        rolstd = pd.Series(self.vec_close).rolling(window=ventana_de_tiempo).std()
        #Plot rolling statistics:
        orig = plt.plot(self.vec_close, color='blue',label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label = 'Rolling Std')
        plt.legend(loc='lower left')
        plt.xticks(rotation=90)
        
        plt.title('Rolling Mean & Standard Deviation')
        plt.show()
    
        #Perform Dickey-Fuller test:
        print ('Results of Dickey-Fuller Test:')
        dftest = adfuller(self.vec_close, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        print (dfoutput)       
    
    
              
    def check_stationarity(self, t = None):
        
        if t is None:
            result = adfuller(self.vec_close)
        else:
            result = adfuller(t)
    
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
    
        if (result[1] <= 0.05) & (result[4]['5%'] > result[0]):
            print("\u001b[32mStationary\u001b[0m")
        else:
            print("\x1b[31mNon-stationary\x1b[0m")
    
   
        plt.show()
    
    
    def correlation(self):
        plt.rcParams.update({'figure.figsize':(12,15),'figure.dpi':120})
        fig, axes = plt.subplots(3,2,sharex= False)
        axes[0, 0].plot(self.vec_close,color='orange'); axes[0, 0].set_title('self.description')
        plot_acf(self.vec_close,ax=axes[0,1],color = 'r')
        
        #Primera diferencia
        axes[1,0].plot(self.vec_close.diff(),color='g'); axes[1,0].set_title('First difference')
        plot_acf(self.vec_close.diff().dropna(),ax=axes[1,1],color = 'r')
        
        #Segunda diferencia
        axes[2,0].plot(self.vec_close.diff().diff(),color='b'); axes[2,0].set_title('Second difference')
        plot_acf(self.vec_close.diff().diff().dropna(),ax=axes[2,1],color = 'r')
        
        plt.show()
        
        

class distribution_manager(): 
    
    def __init__(self, inputs):
        self.inputs = inputs # distribution_inputs
        self.data_table = None
        self.description = None
        self.nb_rows = None
        self.vec_returns = None
        self.vec_close = None
        self.mean = None
        self.std = None
        self.skew = None
        self.kurtosis = None # excess kurtosis
        self.jarque_bera_stat = None # under normality of self.vec_returns this distributes as chi-square with 2 degrees of freedom
        self.p_value = None # equivalently jb < 6
        self.is_normal = None
        self.sharpe = None
        self.var_95 = None
        self.cvar_95 = None
        self.percentile_25 = None
        self.median = None
        self.percentile_75 = None
        
        
    def __str__(self):
        str_self = self.description + ' | size ' + str(self.nb_rows) + '\n' + self.plot_str()
        return str_self
        
        
    def load_timeseries(self):
        
        # data_type = self.inputs['data_type']
        data_type = self.inputs.data_type
        
        if data_type == 'simulation':
            
            nb_sims = self.inputs.nb_sims
            dist_name = self.inputs.variable_name
            degrees_freedom = self.inputs.degrees_freedom
            
            if dist_name == 'normal':
                x = np.random.standard_normal(nb_sims)
                self.description = data_type + ' ' + dist_name
            elif dist_name == 'exponential':
                x = np.random.standard_exponential(nb_sims)
                self.description = data_type + ' ' + dist_name
            elif dist_name == 'uniform':
                x = np.random.uniform(0,1,nb_sims)
                self.description = data_type + ' ' + dist_name
            elif dist_name == 'student':
                x = np.random.standard_t(df=degrees_freedom, size=nb_sims)
                self.description = data_type + ' ' + dist_name + ' | df = ' + str(degrees_freedom)
            elif dist_name == 'chi-square':
                x = np.random.chisquare(df=degrees_freedom, size=nb_sims)
                self.description = data_type + ' ' + dist_name + ' | df = ' + str(degrees_freedom)
       
            self.nb_rows = nb_sims
            self.vec_returns = x
       
        elif data_type == 'real':
           
            
            ric = self.inputs.variable_name
            t = file_function1.load_timeseries(ric)
            
            self.data_table = t
            self.description = 'market data ' + ric
            self.nb_rows = t.shape[0]
            self.vec_returns = t['return_close'].values
            self.vec_close = t['close']
            
    def plot_histogram(self):
       
        plt.figure()
        plt.hist(self.vec_returns,bins=100)
        plt.axvline(x = np.percentile(self.vec_returns,5), c='r', label = "VaR at 95% Confidence Level")
        plt.axvline(x = np.mean(self.vec_returns[self.vec_returns <= self.var_95]), c='r', label = "VaR at 95% Confidence Level")
        plt.title(self.description)
        plt.xlabel(self.plot_str())
        plt.show()
        
    def plot_timeseries(self):
        self.vec_close.plot(figsize=(12,6), fontsize=14)        
        plt.title(self.description)
        plt.xlabel(self.plot_str())
        plt.show()
    
    def compute(self):
        self.mean = np.mean(self.vec_returns)
        self.std = np.std(self.vec_returns)
        self.skew = skew(self.vec_returns)
        self.kurtosis = kurtosis(self.vec_returns) # excess kurtosis
        self.jarque_bera_stat = self.nb_rows/6*(self.skew**2 + 1/4*self.kurtosis**2)
        self.p_value = 1 - chi2.cdf(self.jarque_bera_stat, df=2)
        self.is_normal = (self.p_value > 0.05) # equivalently jb < 6
        self.sharpe = self.mean / self.std * np.sqrt(252) # annualised
        self.var_95 = np.percentile(self.vec_returns,5)
        self.cvar_95 = np.mean(self.vec_returns[self.vec_returns <= self.var_95])
        self.percentile_25 = self.percentile(25)
        self.median = np.median(self.vec_returns)
        self.percentile_75 = self.percentile(75)
    
    def plot_str(self):
        nb_decimals = 4
        plot_str = 'mean ' + str(np.round(self.mean,nb_decimals))\
            + ' | std dev ' + str(np.round(self.std,nb_decimals))\
            + ' | skewness ' + str(np.round(self.skew,nb_decimals))\
            + ' | kurtosis ' + str(np.round(self.kurtosis,nb_decimals)) + '\n'\
            + 'Jarque Bera ' + str(np.round(self.jarque_bera_stat,nb_decimals))\
            + ' | p-value ' + str(np.round(self.p_value,nb_decimals))\
            + ' | is normal ' + str(self.is_normal) + '\n'\
            + 'Sharpe annual ' + str(np.round(self.sharpe,nb_decimals))\
            + ' | VaR 95% ' + str(np.round(self.var_95,nb_decimals))\
            + ' | CVaR 95% ' + str(np.round(self.cvar_95,nb_decimals)) + '\n'\
            + 'percentile 25% ' + str(np.round(self.percentile_25,nb_decimals))\
            + ' | median ' + str(np.round(self.median,nb_decimals))\
            + ' | percentile 75% ' + str(np.round(self.percentile_75,nb_decimals))
        return plot_str
    
    def percentile(self, pct):
        percentile = np.percentile(self.vec_returns,pct)
        return percentile
    
class distribution_input():
    
    def __init__(self):
        self.data_type = None # simulation real custom
        self.variable_name = None # normal student exponential chi-square uniform VWS.CO
        self.degrees_freedom = None # only used in simulation + student and chi-square
        self.nb_sims = None # only in simulation