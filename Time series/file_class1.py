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