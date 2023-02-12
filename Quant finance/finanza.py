# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 19:53:45 2022

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

import file_classes
importlib.reload(file_classes)
import file_functions
importlib.reload(file_functions)


money = 1
interest_rate = 0.05
year = 2
m = 2
time = 0
futuro = money * (1+interest_rate)**year
interest_payments = (money  + (interest_rate/m))**m
frequent  = np.exp(m*np.log(1+(interest_rate/m)))
interest_compuest_after_t = np.exp(interest_rate*time)

def future_value(present_value, annual_rate, years):
    return present_value * (1 + annual_rate / 100) ** years

present_value = 1
annual_rate = 5
years = 2

future_value = future_value(present_value, annual_rate, years)

print("The future value is:", future_value)




def Interes(money): # Cantidad inicial
    return  money* np.exp(interest_rate*year)
Interes(1)

def Valor_presente(money, interest_rate, time, T):
    return money * np.exp(-interest_rate*(T-time))
Valor_presente(1.1051709180756477, 0.05, 0, 2)

random_price = np.random.randint(90,100,100)
plt.figure()
plt.hist(random_price,bins=10)
plt.axvline(x = np.percentile(random_price,50), c='r', label = "VaR at 95% Confidence Level")
#plt.axvline(x = np.mean(random_price[random_price <= np.percentile(random_price,5)], c='r', label = "VaR at 95% Confidence Level")
plt.show()

# Call option
strike = 25
time = 1/12
today = 24.5
profit = max(today-strike,0)

premium = 5

price_expire = 0.50*today + np.linspace(0.1,0.9,100)* 1.1*today
profit = np.array([max(s - strike, 0.0) for s in price_expire]) - Interes(premium)

plt.figure()
plt.title("Options returns")
plt.scatter(price_expire,profit)
plt.axhline(y=0, xmin=0, xmax=1)
plt.xlabel("Price")
plt.ylabel('Profit')
plt.show()

prob = profit > 0
pro_profit = np.mean(prob)

Interes(premium)


### EJEMPLO
def payoff(strike_1,strike_2,today):
    
    price_expire = 0.8*today + np.linspace(0.1,0.9,100)* 1.2*today
    payoff = (1/(strike_2- strike_1)) * (np.array([max(s - strike_1, 0.0) for s in price_expire])- np.array([max(s - strike_2, 0.0) for s in price_expire]))
    
    return payoff


strike_1 = 100
strike_2 = 120
today = 90
hola = payoff(strike_1, strike_2,today)

plt.figure()
plt.title("Options payoff")
plt.scatter(price_expire,hola)
plt.xlabel("Price")
plt.ylabel('Payoff')
plt.show()

aluar = file_functions.load_timeseries('ALUA.BA')
m = len(aluar['return_close'])
aluar_mean = np.mean(aluar['close'])
aluar_std = np.sqrt((1/(m-1))*sum((aluar['close']-aluar_mean)**2))
scipy.stats.norm.cdf(aluar_std)
aluar['return_close'][0]*np.exp(aluar_mean*1/252)
vol = np.sqrt((1/(m-1))*sum((np.log(aluar['close'])-np.log(aluar['close_previous']))**2))
rand = np.random.randint(1,4,12)
next1 = aluar['close'][241]*(1+aluar_mean*(1/252)+vol*rand[0]*(1/252)**1/2)


from random import randint
def CoinToss(number):
    flips = [randint(0,1) for r in range(number)]
    yo = []
    you = []
    CoinToss.si = []
    results = []
    
    for object in flips:
        if object == 0:
            yo.append(1)
            CoinToss.si.append(1)
            
            results.append('Heads')
        elif object == 1:
            you.append(1)
            CoinToss.si.append(-1)
            results.append('Tails')
            
        
    return results  ,sum(yo),sum(you)
CoinToss(10)


a = CoinToss.si
b = 1 - np.mean(a)
np.mean(a)

def quadratic_variation(random_walk, t0, tn):
  """
  Computes the quadratic variation of the given random walk over the time interval [t0, tn].

  Args:
    random_walk: a list or array of values representing the random walk
    t0: the starting time (inclusive) of the time interval
    tn: the ending time (exclusive) of the time interval

  Returns:
    The quadratic variation of the random walk over the time interval [t0, tn]
  """
  qv = 0
  for i in range(t0, tn-1):
    qv += (random_walk[i+1] - random_walk[i])**2
  return qv



random_walk = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
qv = quadratic_variation(random_walk, 2, 5)
print(qv)  # prints "14"


import numpy as np

def ito_integral(process, t0, tn, N):
  """
  Computes the Itô integral of the given process over the time interval [t0, tn], using the trapezoidal method.

  Args:
    process: a function representing the process to be integrated
    t0: the starting time (inclusive) of the time interval
    tn: the ending time (exclusive) of the time interval
    N: the number of subintervals to use in the approximation

  Returns:
    The Itô integral of the process over the time interval [t0, tn]
  """
  # Compute the size of the subintervals
  dt = (tn - t0) / N
  print(dt)
  # Compute the values of the process at the start and end of each subinterval
  t = np.linspace(t0, tn, N+1)
  print(t)
  X = process(t)
  
  # Compute the weighted average of the process at each subinterval
  X_avg = (X[:-1] + X[1:]) / 2
  print(X_avg)

  # Compute the Itô integral as the sum of the weighted averages, multiplied by the subinterval size
  integral = np.sum(X_avg * dt)
  return integral

def process(t):
  return t**2

result = ito_integral(process, 0, 3, 10)
print(result)

import matplotlib.pyplot as plt

# Genera una secuencia equiespaciada de puntos entre 0 y 3 con 4 subintervalos
X = np.linspace(0, 3, num=5)

# Calcula el valor del proceso aleatorio en cada punto de la secuencia
Y = process(X)

# Grafica el proceso aleatorio
plt.plot(X, Y)
plt.show()




def brownian_motion_with_drift(t0, tn, dt, drift, initial_value, sigma):
  """
  Genera un movimiento browniano con un sesgo.

  Parameters:
  - t0: tiempo inicial
  - tn: tiempo final
  - dt: tamaño del subintervalo
  - drift: tasa de sesgo
  - initial_value: valor inicial
  - sigma: volatilidad

  Returns:
  - Una secuencia equiespaciada de puntos que representan el movimiento browniano con un sesgo.
  """
  # Genera una secuencia equiespaciada de puntos entre t0 y tn con dt como tamaño de subintervalo
  t = np.linspace(t0, tn, int((tn - t0) / dt) + 1)
  
  # Genera una secuencia de números aleatorios normales con media 0 y desviación estándar 1
  rand = np.random.normal(0, 1, size=t.shape)
  
  # Calcula el movimiento browniano con un sesgo
  y = initial_value + drift * t + sigma * rand
  
  return y


Z = brownian_motion_with_drift(0, 4, 0.75, 0.5, 0, 0.8)


# Genera una secuencia equiespaciada de puntos entre 0 y 3 con 4 subintervalos
X = np.linspace(0, 4, num=6)

# Calcula el valor del movimiento browniano con un sesgo en cada punto de la secuencia
Y = brownian_motion_with_drift(0, 3, 0.75, 0.5, 0, 1)

# Grafica el movimiento browniano con un sesgo
plt.plot(X, Z)
plt.show()

c_option = option + change
c_asset = asset + change_o
delta  = c_option / c_asset
