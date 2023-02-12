# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 08:47:37 2023

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
import math

import random





# Get the stock info for Apple
apple = yf.Ticker("AAPL")

# Get the historical prices for the past 2 years
historical_prices = apple.history(period="2y")

# Get the closing prices
closing_prices = historical_prices['Close']
returns = closing_prices.pct_change().dropna()
anual_returns = returns.mean() * 252
annualized_std_dev = returns.std() * math.sqrt(252)


print(closing_prices)


def monte_carlo_option_pricer(S, K, r, sigma, T, num_sims):
    """
    Calculates the price of an option using the Monte Carlo method.
    
    S: current stock price
    K: strike price
    r: risk-free interest rate
    sigma: volatility of the stock
    T: time to expiration (in years)
    num_sims: number of simulations to run
    
    Returns: the price of the option
    """
    total_payoff = 0
    
    for i in range(num_sims):
        S_T = S * math.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * random.gauss(0, 1))
        payoff = max(0, S_T - K)
        total_payoff += payoff
    
    return total_payoff / num_sims * math.exp(-r * T)


S = 125 # current stock price of Apple
K = 135 # strike price of the option
r = 0.0379 # risk-free interest rate
sigma = annualized_std_dev # volatility of Apple stock
T = 1 # time to expiration of the option
num_sims = 10**6 # number of simulations to run

option_price = monte_carlo_option_pricer(S, K, r, sigma, T, num_sims)
p= 20
b =+ p
for i in range(10):
    b += b
    print(b)
    
