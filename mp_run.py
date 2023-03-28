from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
import numpy as np
from xgboost import XGBRegressor
from xgboost import XGBRFRegressor


import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore")

perturbation_factor = 3

def rf_feature_importance(ts_train_X, ts_train_y, perturbation_input, perturbation_input_alt, i):
    # regr = XGBRegressor(random_state=i, n_jobs=1)
    regr = RandomForestRegressor(random_state=i, n_jobs=1)
    # regr = XGBRFRegressor(random_state=i, n_jobs=1)
    # regr = AdaBoostRegressor(random_state=i)
    regr = regr.fit(ts_train_X, ts_train_y)
    input_mean = ts_train_X.mean()
    base_predict = regr.predict(np.array(input_mean).reshape(1,-1))[0]
    return [base_predict - regr.predict(perturbation_input), base_predict - regr.predict(perturbation_input_alt)]