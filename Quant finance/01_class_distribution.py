# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 12:56:27 2022

@author: ilarr
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2

import file_classes
importlib.reload(file_classes)

import file_classes
importlib.reload(file_classes)



inputs = file_classes.distribution_input()
inputs.data_type = 'real' # real simulation
inputs.variable_name = 'ALUA.BA' # TXR.BA , ALUA.BA , ^MERV
inputs.degrees_freedom = None
inputs.nb_sims = None

dm = file_classes.distribution_manager(inputs) # initialise constructor
dm.load_timeseries() # get the timeseries
dm.compute() # compute returns and all different risk metrics
dm.plot_histogram() # plot histogram
print(dm) # write all data in console
dm.percentile(80)
            

