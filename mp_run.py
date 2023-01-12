from sklearn.ensemble import RandomForestRegressor
import numpy as np

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore")

perturbation_factor = 3

def rf_feature_importance(ts_train_X, ts_train_y, i):
    regr = RandomForestRegressor(random_state=i, warm_start=True, n_estimators=300, n_jobs=1)
    regr = regr.fit(ts_train_X, ts_train_y)
    return regr
    input_mean = ts_train_X.mean()
    input_std = ts_train_X.std()
    perturbation_importance = []
    base_prediction = regr.predict(np.array(input_mean).reshape(1,-1))[0]
    for gene in ts_train_X.columns:
        perturbation_input = input_mean.copy()
        perturbation_input[gene] += input_std[gene]
        perturbation_prediction = regr.predict(np.array(perturbation_input).reshape(1,-1))[0]
        perturbation_importance.append(np.abs((perturbation_prediction - base_prediction)/ts_train_y.std()))
    return perturbation_importance