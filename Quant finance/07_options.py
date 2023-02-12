# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 21:16:57 2022

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

# import our own files and reload
import file_classes
importlib.reload(file_classes)
import file_functions
importlib.reload(file_functions)

# inputs
inputs = file_classes.option_input()
inputs.price = 125
inputs.time = 0 # in years
inputs.volatility = 0.3093343187645484
inputs.interest_rate = 0.0379
inputs.maturity = 1 # in years
inputs.strike = 135
inputs.call_or_put = 'call'
number_simulations = 1*10**6

# price using Black-Scholes formula
price_black_scholes = file_functions.compute_price_black_scholes(inputs)

# price using Monte Carlo simulations
prices_monte_carlo = file_functions.compute_price_monte_carlo(inputs, number_simulations)
print(prices_monte_carlo)
prices_monte_carlo.plot_histogram()
