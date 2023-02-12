# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 11:58:07 2022

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

# import our own files and reload
import file_classes
importlib.reload(file_classes)
import file_functions
importlib.reload(file_functions)

# inputs
inputs = file_classes.hedge_input()
inputs.benchmark = '^MERV'
inputs.security = 'TXR.BA'
inputs.hedge_securities =  ['ALUA.BA','CRES.BA','PAMP.BA'] #'ALUA.BA', TXR.BA','CRES.BA','PAMP.BA'
inputs.delta_portfolio = 10 # mn USD

# computations
hedge = file_classes.hedge_manager(inputs)
hedge.load_betas() # get the betas for portfolio and hedges
hedge.compute(regularisation=0.1) # numerical solution
hedge_delta = hedge.hedge_delta
hedge_beta_usd = hedge.hedge_beta_usd