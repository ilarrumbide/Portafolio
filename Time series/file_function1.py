# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 19:45:53 2023

@author: ilarr
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2, linregress
from scipy.optimize import minimize
from numpy import linalg as LA
import yfinance as yf

# import our own files and reload
import file_class1
importlib.reload(file_class1)
import file_function1
importlib.reload(file_function1)

                 
def load_timeseries(ric):
    
    # Get the data for the asset
    asset = yf.Ticker(ric)
   
   # Get the historical prices for the past year
    raw_data = asset.history(period="1y")
    raw_data = raw_data.reset_index()
    t = pd.DataFrame()
    t['date'] = pd.to_datetime(raw_data['Date'], dayfirst=True)
    t['close'] = raw_data['Close']
    t.sort_values(by='date', ascending=True)
    t['close_previous'] = t['close'].shift(1)
    t['return_close'] = t['close']/t['close_previous'] - 1
    q_low =t["return_close"].quantile(0.001)
    q_hi  = t["return_close"].quantile(0.999)
    
    t = t[(t["return_close"] < q_hi) & (t["return_close"] > q_low)  ]
    
    t = t.dropna()
    t = t.reset_index(drop=True)
    
    return t

def split_ts(data,train_percent = None,test_percent=None):
    if train_percent is None:
        train = int(len(data)*0.8)
    else:
        train = int(len(data)*train_percent)
    if test_percent is None:
        test = int(len(data)*0.2)
    else:
        test = int(len(data)*test_percent)
    
    return train, test


def split_ts(data,train_percent = None,test_percent=None):
    if train_percent is None:
        train_percent = 0.8
    if test_percent is None:
        test_percent = 0.2

    train_len = int(len(data)*train_percent)
    test_len = int(len(data)*test_percent)

    train_data = data[:train_len]
    test_data = data[train_len:]

    return train_data, test_data

def get_train_test(split_data):
    return split_data[0], split_data[1]




    

