# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 20:28:04 2022

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
import matplotlib.pyplot as plt

# import our own files and reload
import file_classes
importlib.reload(file_classes)
import file_functions
importlib.reload(file_functions)

rics = ['ALUA.BA','CRES.BA','PAMP.BA','TXR.BA']

notional = 300
t = file_functions.table(rics,notional,table = 'yes')
target_return = 0.895
include_min_variance=False
dict_portfolios = file_functions.compute_efficient_frontier(rics, notional, target_return, include_min_variance)

 # compute vectors of returns and volatilities for Markowitz portfolios
def best_portafolios(t,tabla = pd.DataFrame()):
    best = 10
    volatilidad = np.zeros([10,1])
    retornos = np.zeros([10,1])
    cartera = {}
    counter = 0
    for i in range(best):
        volatilidad[counter]= list(t.iloc[i].volatilities)
        retornos[counter] = list(t.iloc[i].returns)
        cartera[counter] = (t.iloc[i].type)
        counter += 1
    
    tabla['volatilidad'] =pd.DataFrame(volatilidad)
    tabla['retornos'] = retornos
    tabla['cartera'] = cartera
    
  
        
    plt.figure()
    plt.title('Efficient Frontier for a portfolio including ' + rics[0])
    plt.scatter(volatilidad,retornos)
    plt.plot(volatilidad[0], retornos[0], "sy", label=cartera[0]) # yellow triangle
    plt.plot(volatilidad[1], retornos[1], "ok", label=cartera[1]) # black cross
    plt.plot(volatilidad[2], retornos[2], "^r", label=cartera[2]) # red dot
    plt.plot(volatilidad[3], retornos[3], "^y", label=cartera[3]) # yellow square
    plt.plot(volatilidad[4], retornos[4], "^k", label=cartera[4]) # black square

    plt.legend(loc='best',  borderaxespad=0.)
     
    plt.ylabel('portfolio return')
    plt.xlabel('portfolio volatility')
    plt.grid()

    plt.show()   
    return tabla
best = 10    
tabla = best_portafolios(t)   


