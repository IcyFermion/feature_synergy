from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from xgboost import XGBRFRegressor
from sklearn.model_selection import train_test_split, cross_val_score

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore")

class MpCalc:
    def __init__(self, target_gene_list, target_exp, X, perturbation_factor=3):
        self.target_gene_list = target_gene_list
        self.target_exp = target_exp
        self.X = X
        self.perturbation_factor = perturbation_factor
        # self.network_df = network_df
        # self.train_source = train_source
        # self.train_target = train_target
        # self.test_source = test_source
        # self.test_target = test_target

    def calc(self, index):
        target = self.target_gene_list[index]
        y = self.target_exp.loc[target]
        scores = []
        for i in range(20):
            regr = RandomForestRegressor(random_state=42, n_jobs=1, oob_score=True, max_features='sqrt')
            regr.fit(self.X.T, y)
            scores.append(regr.oob_score_)
        return np.array([np.mean(scores), np.std(scores)])
    
    def double_perturb(self, index):
        target_index = 0
        y = self.target_exp.iloc[index]

        regr_list = []
        for i in range(20):
            regr = RandomForestRegressor(n_jobs=1, random_state=i**2, max_features='sqrt')
            regr.fit(self.X.T, y)
            regr_list.append(regr)

        tf_mean = self.X.mean(axis=1)
        tf_std = self.X.std(axis=1)
        reference_input = pd.concat([tf_mean]*len(tf_mean), axis=1)

        single_perturb_input = reference_input.copy()
        for i in range(len(tf_std)):
            single_perturb_input[i][i] += tf_std[i] * self.perturbation_factor

        single_effect_list = []
        for i in range(20):
            single_effect_list.append(regr_list[i].predict(single_perturb_input.T) - regr_list[i].predict(reference_input.T))
        single_effect_list = np.array(single_effect_list)
        single_effect = single_effect_list.mean(axis=0)
        single_effect_sorted_arg = np.argsort(single_effect)

        activator_top = single_effect_sorted_arg[-5:]
        repressor_top = single_effect_sorted_arg[:5]

        double_effect_input = pd.concat([tf_mean]*45, axis=1)
        double_effect_input_list = []
        double_compare_list = []
        for i in range(0, 5):
            for j in range(i+1, 5):
                tf_i = activator_top[i]
                tf_j = activator_top[j]
                new_input = tf_mean.copy()
                new_input[tf_i] += tf_std[tf_i] * self.perturbation_factor
                new_input[tf_j] += tf_std[tf_j] * self.perturbation_factor
                double_effect_input_list.append(new_input)
                double_compare_list.append(np.max([single_effect[tf_i], single_effect[tf_j]]))

        for i in range(0, 5):
            for j in range(i+1, 5):
                tf_i = repressor_top[i]
                tf_j = repressor_top[j]
                new_input = tf_mean.copy()
                new_input[tf_i] += tf_std[tf_i] * self.perturbation_factor
                new_input[tf_j] += tf_std[tf_j] * self.perturbation_factor
                double_effect_input_list.append(new_input)
                double_compare_list.append(np.min([single_effect[tf_i], single_effect[tf_j]]))

        for i in range(0, 5):
            for j in range(0, 5):
                tf_i = activator_top[i]
                tf_j = repressor_top[j]
                new_input = tf_mean.copy()
                new_input[tf_i] += tf_std[tf_i] * self.perturbation_factor
                new_input[tf_j] += tf_std[tf_j] * self.perturbation_factor
                double_effect_input_list.append(new_input)
                double_compare_list.append(np.min([single_effect[tf_i], single_effect[tf_j]]))
        double_perturb_input = pd.concat(double_effect_input_list, axis=1)

        double_effect_res_list = []
        for i in range(20):
            double_effect_res_list.append(regr_list[i].predict(double_perturb_input.T) - regr_list[i].predict(tf_mean.values.reshape(1,-1)))
        double_effect_res_list = np.array(double_effect_res_list)

        return np.array([double_effect_res_list.mean(axis=0), np.array(double_compare_list)])

