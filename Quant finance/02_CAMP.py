# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 12:35:23 2022

@author: ilarr
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2, linregress

# import our own files and reload
import file_classes
importlib.reload(file_classes)
import file_functions
importlib.reload(file_functions)

# inputs
benchmark = '^MERV' # variable x
security = 'ALUA.BA' # variable y ,'TXR.BA','ALUA.BA'

capm = file_classes.capm_manager(benchmark, security)
capm.load_timeseries()
# capm.plot_timeseries()
capm.compute()
capm.plot_timeseries()
capm.plot_linear_regression()

print(capm)