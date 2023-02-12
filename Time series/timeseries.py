# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 10:53:34 2023

@author: ilarr
"""
import importlib
import file_class1
importlib.reload(file_class1)

from sklearn.metrics import mean_squared_error, mean_absolute_error

import file_function1
importlib.reload(file_function1)
import file_class1
importlib.reload(file_class1)


inputs = file_class1.distribution_input()
inputs.data_type = 'real' # real simulation
inputs.variable_name = 'BTC-USD' # TXR.BA , ALUA.BA , ^MERV
inputs.degrees_freedom = None
inputs.nb_sims = None

tm = file_class1.time_series(inputs)
t = tm.load_timeseries()

L = file_function1.split_ts(t,0.98,0.02)
train_data, test_data = file_function1.get_train_test(L)
tsm = file_class1.TimeSeriesModel(train_data.close, test_data.close)

best_order, seasonal_order = tsm.find_best_order()

results, resid = tsm.fit_model(best_order,seasonal_order)

tsm.print_summary()
tsm.check_residuals()
tsm.plot_residuals()
forecast = (tsm.forecast())
tsm.plot_forecast(forecast,100)
tsm.compare_forecast(forecast.predict)


forecast_mean, forecast_conf_int, residuals = tsm.forecast_future(steps = 20,order=best_order,season=seasonal_order)
tsm.plot_future_forecast(forecast_mean, forecast_conf_int)



mse = mean_squared_error(test_data.close, forecast.predict)
mae = mean_absolute_error(test_data.close, forecast.predict)
print("MSE: ", mse)
print("MAE: ", mae)

