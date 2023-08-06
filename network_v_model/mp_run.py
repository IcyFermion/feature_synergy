from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
import numpy as np
from xgboost import XGBRegressor
from xgboost import XGBRFRegressor
from sklearn.model_selection import train_test_split, cross_val_score

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore")

class MpCalc:
    def __init__(self, target_gene_list, target_exp, X):
        self.target_gene_list = target_gene_list
        self.target_exp = target_exp
        self.X = X
    
    def calc(self, index):
        target = self.target_gene_list[index]
        y = self.target_exp.loc[target]
        xb_regr = RandomForestRegressor(random_state=42, n_jobs=1)
        scores = cross_val_score(xb_regr, self.X.T, y, cv=5)
        return np.mean(scores)
