# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:05:17 2022

@author: ilarr
"""
# IMPORTAR LIBRERIAS
import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2

#import file_classes
#importlib.reload(file_classes)
#import file_functions
#importlib.reload(file_functions)
# Crear tabla


data = pd.read_csv('^MERV.csv')
t = pd.DataFrame()
t['date'] = pd.to_datetime(data['Date'],dayfirst = True)
t['close'] = data['Close']
t.sort_values(by = 'date', ascending = True)
t['return'] = t['close'].pct_change()
t = t.dropna()
t = t.reset_index(drop = True)
print(t['return'].mean())
std_daily = t['return'].std()
print('daily vol: ','{:.2f}%'.format(std_daily))
print('monthly vol: ','{:.2f}%'.format(std_daily*np.sqrt(21)))
print('year vol: ','{:.2f}%'.format(std_daily*np.sqrt(252)))

x = t['return'].values
x_description = 'Aluar' 
plt.figure()
plt.hist(x,bins=20)
plt.title(x_description)
plt.show()

plt.figure()
plt.plot(t['date'],t['close'])
plt.title("Aluar serie de tiempo")
plt.xlabel('tiempo')
plt.ylabel('precio')
plt.show()