from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
import numpy as np
from xgboost import XGBRegressor
from xgboost import XGBRFRegressor
from sklearn.model_selection import train_test_split


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


def regr_perturbation(ts_train_X, ts_train_y, perturbation_input, perturbation_input_alt, i):
    regr = XGBRegressor(random_state=i, n_jobs=1)
    # regr = RandomForestRegressor(random_state=i, n_jobs=1)
    # regr = XGBRFRegressor(random_state=i, n_jobs=1)
    # regr = AdaBoostRegressor(random_state=i)
    regr = regr.fit(ts_train_X, ts_train_y)
    input_mean = ts_train_X.mean()
    base_predict = regr.predict(np.array(input_mean).reshape(1,-1))[0]
    return [regr.predict(perturbation_input) - base_predict, regr.predict(perturbation_input_alt) - base_predict]


def validation_measure(ts_train_X, ts_train_y, i):
    # regr = XGBRegressor(random_state=i, n_jobs=1)
    regr = RandomForestRegressor(random_state=i, n_jobs=1)
    # regr = XGBRFRegressor(random_state=i, n_jobs=1)
    # regr = AdaBoostRegressor(random_state=i)

    X_train, X_test, y_train, y_test = train_test_split(ts_train_X, ts_train_y, test_size=0.33, random_state=i*i)
    regr = regr.fit(X_train, y_train)
    return regr.score(X_test, y_test)