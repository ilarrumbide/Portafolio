# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 18:03:24 2023

@author: ilarr
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import importlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import power_transform


import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from statsmodels.tsa.statespace.sarimax import SARIMAX


import importlib
import file_class1
importlib.reload(file_class1)

import file_function1
importlib.reload(file_function1)

inputs = file_class1.distribution_input()
inputs.data_type = 'real' # real simulation
inputs.variable_name = 'BTC-USD' # TXR.BA , ALUA.BA , ^MERV
inputs.degrees_freedom = None
inputs.nb_sims = None


dm = file_class1.distribution_manager(inputs) # initialise constructor
dm.load_timeseries() # get the timeseries
dm.compute() # compute returns and all different risk metrics
dm.plot_histogram() # plot histogram
print(dm) # write all data in console
dm.percentile(80)
dm.plot_timeseries()    



tm = file_class1.time_series(inputs)
t = tm.load_timeseries()
tm.plot_timeseries()

tm.descomposicion()
tm.test_stationarity()
tm.check_stationarity(t.close.diff().diff().dropna())
tm.descomposicion(t.close.diff().diff().dropna())
tm.correlation()


L = file_function1.split_ts(t,0.95,0.05)
train_data, test_data = file_function1.get_train_test(L)

