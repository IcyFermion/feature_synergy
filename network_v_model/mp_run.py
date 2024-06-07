from tqdm import tqdm
import time
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn import linear_model
import numpy as np
import pandas as pd
# from xgboost import XGBRegressor
# from xgboost import XGBRFRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import explained_variance_score, mean_squared_error

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from scipy import stats

from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings("ignore")

MINIMAL_SET_SIZE_LIMIT = 10

class MpCalc:
    def __init__(self, target_gene_list, tf_list, network_df, train_source, train_target, test_source, test_target):
        self.target_gene_list = target_gene_list
        self.network_df = network_df
        self.train_source = train_source
        self.train_target = train_target
        self.test_source = test_source
        self.test_target = test_target
        self.available_tfs = set(tf_list)
        self.tf_list = list(tf_list)

    def get_train_test_sets(self, index):
        target = self.target_gene_list[index]
        gs_tf_list = self.network_df.loc[target].tf_list
        gs_tf_list = gs_tf_list.split('; ')
        gs_tf_list = list(self.available_tfs.intersection(gs_tf_list))
        if target in gs_tf_list: gs_tf_list.remove(target)
        if target in self.train_source.loc[self.tf_list].index:
            train_X = self.train_source.loc[self.tf_list].drop([target])
            test_X = self.test_source.loc[self.tf_list].drop([target])
        else:
            train_X = self.train_source.loc[self.tf_list]
            test_X = self.test_source.loc[self.tf_list]
        y_train = self.train_target.loc[target]
        y_test = self.test_target.loc[target]
        return train_X, test_X, y_train, y_test, gs_tf_list

    def lime_test(self, index):
        target = self.target_gene_list[index]
        tf_list = self.network_df.loc[target].tf_list
        tf_list = tf_list.split('; ')
        tf_list = list(self.available_tfs.intersection(tf_list))
        y_train = self.train_target.loc[target]
        y_test = self.test_target.loc[target]
        rf_regr = RandomForestRegressor(random_state=42, n_jobs=1)
        network_rf_regr = RandomForestRegressor(random_state=42, n_jobs=1)
        linear_regr = LinearRegression()
        new_linear_regr = LinearRegression()
        rf_regr.fit(self.train_source.T, y_train)
        network_rf_regr.fit(self.train_source.loc[tf_list].T, y_train)
        linear_regr.fit(self.train_source.loc[tf_list].T, y_train)
        top_rf_tf_list = np.flip(rf_regr.feature_names_in_[np.argsort(rf_regr.feature_importances_)[-10:]])
        new_linear_regr.fit(self.train_source.loc[top_rf_tf_list].T, y_train)
        return np.array([
            rf_regr.score(self.test_source.T, y_test),
            linear_regr.score(self.test_source.loc[tf_list].T, y_test),
            new_linear_regr.score(self.test_source.loc[top_rf_tf_list].T, y_test),
            network_rf_regr.score(self.test_source.loc[tf_list].T, y_test)
            ])

    def efron_process_rf(self, index):
        return self.efron_process(index, 'rf')
    def efron_process_linear(self, index):
        return self.efron_process(index, 'linear')

    def efron_process(self, index, regr_type):
        train_X, test_X, y_train, y_test, tf_list = self.get_train_test_sets(index)
        if (regr_type == 'rf'):
            regr = RandomForestRegressor(random_state=42, n_jobs=1, max_features='sqrt' )
        else:
            regr = linear_model.Ridge()
        regr.fit(train_X.T, y_train)
        base_error = np.square(regr.predict(test_X.T)-y_test)
        feature_list = train_X.index
        if (regr_type == 'rf'):
            current_feature_importance = regr.feature_importances_
        else:
            current_feature_importance = np.abs(regr.coef_) / np.sum(np.abs(regr.coef_))
        current_rmse = mean_squared_error(
            regr.predict(test_X.T),
            y_test, squared=False
        )
        continue_flag = True
        while(continue_flag):
            half_feature_num = int(len(feature_list)/2)
            top_half_features = feature_list[np.argsort(current_feature_importance)[-1*half_feature_num:]]
            current_train_X = train_X.loc[top_half_features]
            current_test_X = test_X.loc[top_half_features]
            if (regr_type == 'rf'):
                current_regr = RandomForestRegressor(random_state=42, n_jobs=1, max_features='sqrt' )
            else:
                current_regr = linear_model.Ridge()
            current_regr.fit(current_train_X.T, y_train)
            current_error = np.square(current_regr.predict(current_test_X.T)-y_test)
            s, p = stats.ttest_rel(base_error, current_error)
            current_score = current_regr.score(current_test_X.T, y_test)
            if (current_score < 0):
                continue_flag = False
                break
            if (s < 0) and (p < 0.05):
                continue_flag = False
                break
            feature_list = top_half_features
            current_rmse = mean_squared_error(
                current_regr.predict(current_test_X.T),
                y_test, squared=False
            )
            if (regr_type == 'rf'):
                current_feature_importance = current_regr.feature_importances_
            else:
                current_feature_importance = np.abs(current_regr.coef_) / np.sum(np.abs(current_regr.coef_))

            if (len(feature_list) < 4):
                continue_flag = False
                break

        complementary_features_rmse_list = []
        complementary_features_length_list = []
        complementary_features_idx_list = []
        left_over_features = train_X.index.difference(feature_list)
        complementary_features = left_over_features
        while (len(left_over_features) > 0):
            complementary_features = left_over_features
            current_train_X = train_X.loc[complementary_features]
            current_test_X = test_X.loc[complementary_features]
            if (regr_type == 'rf'):
                current_regr = RandomForestRegressor(random_state=42, n_jobs=1, max_features='sqrt' )
            else:
                current_regr = linear_model.Ridge()
            current_regr.fit(current_train_X.T, y_train)
            current_error = np.square(current_regr.predict(current_test_X.T)-y_test)
            if (regr_type == 'rf'):
                current_feature_importance = current_regr.feature_importances_
            else:
                current_feature_importance = np.abs(current_regr.coef_) / np.sum(np.abs(current_regr.coef_))
            s, p = stats.ttest_rel(base_error, current_error)
            current_score = current_regr.score(current_test_X.T, y_test)
            if (current_score < 0):
                continue_flag = False
                complementary_rmse = np.iinfo(np.int16).max
                break
            if (s < 0) and (p < 0.05):
                continue_flag = False
                complementary_rmse = np.iinfo(np.int16).max
                break
            else:
                continue_flag = True
                complementary_rmse = mean_squared_error(
                    current_regr.predict(test_X.loc[complementary_features].T),
                    y_test, squared=False
                )
            while (continue_flag):
                half_feature_num = int(len(complementary_features)/2)
                top_half_features = complementary_features[np.argsort(current_feature_importance)[-1*half_feature_num:]]
                current_train_X = train_X.loc[top_half_features]
                current_test_X = test_X.loc[top_half_features]
                if (regr_type == 'rf'):
                    current_regr = RandomForestRegressor(random_state=42, n_jobs=1, max_features='sqrt' )
                else:
                    current_regr = linear_model.Ridge()
                current_regr.fit(current_train_X.T, y_train)
                current_error = np.square(current_regr.predict(current_test_X.T)-y_test)
                s, p = stats.ttest_rel(base_error, current_error)
                current_score = current_regr.score(current_test_X.T, y_test)
                if (current_score < 0):
                    continue_flag = False
                    # print('bad...')
                    break
                if (s < 0) and (p < 0.05):
                    continue_flag = False
                    # print('bad...')
                    break
                complementary_features = top_half_features
                complementary_rmse = mean_squared_error(
                    current_regr.predict(current_test_X.T),
                    y_test, squared=False
                )
                if (regr_type == 'rf'):
                    current_feature_importance = current_regr.feature_importances_
                else:
                    current_feature_importance = np.abs(current_regr.coef_) / np.sum(np.abs(current_regr.coef_))

                if (len(complementary_features) < 4):
                    continue_flag = False
                    break
            complementary_features_index = [train_X.index.get_loc(feature) for feature in complementary_features]
            complementary_features_rmse_list.append(complementary_rmse)
            complementary_features_idx_list.append('; '.join(str(v) for v in complementary_features_index))
            complementary_features_length_list.append(len(complementary_features_index))
            left_over_features = left_over_features.difference(complementary_features)
            # print(complementary_features_index)
            # print(current_score)
            # print(s,p)

        # else: complementary_rmse = np.iinfo(np.int16).max
        if (len(complementary_features_rmse_list) < 1):
            complementary_features_rmse_list.append(np.iinfo(np.int16).max)
        feature_list_index = [train_X.index.get_loc(feature) for feature in feature_list]
        # complementary_features_index = [train_X.index.get_loc(feature) for feature in complementary_features]

        return np.array([
            len(feature_list),
            ', '.join(str(v) for v in complementary_features_length_list),
            current_rmse,
            ', '.join(str(v) for v in complementary_features_rmse_list),
            '; '.join(str(v) for v in feature_list_index),
            ', '.join(str(v) for v in complementary_features_idx_list)
            ])

    def efron_process_rf_training(self, index):
        return self.efron_process_training(index, 'rf')
    def efron_process_linear_training(self, index):
        return self.efron_process_training(index, 'linear')

    def efron_process_training(self, index, regr_type):
        train_X, test_X, y_train, y_test, tf_list = self.get_train_test_sets(index)
        # function for random comparison
        need_for_random_comparision = True
        random_comparison_tuple = (0,0)
        def check_for_random_comparison(minimal_set_size, 
                                        minimal_set_rmse, 
                                        train_X=train_X, 
                                        test_X=test_X,
                                        need_for_random_comparision=need_for_random_comparision,
                                        random_comparison_tuple=random_comparison_tuple
                                       ):
            if (minimal_set_size < MINIMAL_SET_SIZE_LIMIT) and need_for_random_comparision:
                random_features = train_X.sample(n=minimal_set_size, random_state=43).index
                random_train_X = train_X.loc[random_features]
                random_test_X = test_X.loc[random_features]
                random_regr = RandomForestRegressor(random_state=42, n_jobs=1, max_features='sqrt' )
                random_regr.fit(random_train_X.T, y_train)
                random_rmse = mean_squared_error(
                    random_regr.predict(random_test_X.T),
                    y_test, squared=False
                )
                random_comparison_tuple = (minimal_set_rmse, random_rmse)
                need_for_random_comparision = False
            return need_for_random_comparision, random_comparison_tuple
        y_pred_ref = self.test_source.loc[self.target_gene_list[index]]
        ref_squared_error = np.square(y_pred_ref.values-y_test.values)
        if (regr_type == 'rf'):
            regr = RandomForestRegressor(random_state=42, n_jobs=1, max_features='sqrt' )
        else:
            regr = linear_model.Ridge()
        regr.fit(train_X.T, y_train)
        base_error = np.square(regr.predict(train_X.T)-y_train)
        feature_list = train_X.index
        if (regr_type == 'rf'):
            current_feature_importance = regr.feature_importances_
        else:
            current_feature_importance = np.abs(regr.coef_) / np.sum(np.abs(regr.coef_))
        current_rmse = mean_squared_error(
            regr.predict(test_X.T),
            y_test, squared=False
        )
        continue_flag = True
        current_predict = regr.predict(test_X.T)
        while(continue_flag):
            half_feature_num = int(len(feature_list)/2)
            top_half_features = feature_list[np.argsort(current_feature_importance)[-1*half_feature_num:]]
            current_train_X = train_X.loc[top_half_features]
            current_test_X = test_X.loc[top_half_features]
            if (regr_type == 'rf'):
                current_regr = RandomForestRegressor(random_state=42, n_jobs=1, max_features='sqrt' )
            else:
                current_regr = linear_model.Ridge()
            current_regr.fit(current_train_X.T, y_train)
            current_error = np.square(current_regr.predict(current_train_X.T)-y_train)
            s, p = stats.ttest_rel(base_error, current_error)
            current_score = current_regr.score(current_train_X.T, y_train)
            if (current_score < 0):
                continue_flag = False
                break
            if (s < 0) and (p < 0.05):
                continue_flag = False
                break
            feature_list = top_half_features
            current_rmse = mean_squared_error(
                current_regr.predict(current_test_X.T),
                y_test, squared=False
            )
            current_predict = current_regr.predict(current_test_X.T)
            if (regr_type == 'rf'):
                current_feature_importance = current_regr.feature_importances_
            else:
                current_feature_importance = np.abs(current_regr.coef_) / np.sum(np.abs(current_regr.coef_))

            if (len(feature_list) < 4):
                continue_flag = False
                break

        complementary_features_rmse_list = []
        complementary_features_length_list = []
        complementary_features_idx_list = []
        ensemble_predict_list = [current_predict]
        filtered_minimal_set_list = []
        filtered_minimal_set_rmse_list = []

        current_squared_error = np.square(current_predict-y_test)

        # comparing with repeating value
        ref_comp_ttest = stats.ttest_rel(current_squared_error, ref_squared_error)
        repeating_value_comp_diff = [ref_comp_ttest.statistic]
        repeating_value_comp_pval = [ref_comp_ttest.pvalue]
        if ref_comp_ttest.statistic < 0 and ref_comp_ttest.pvalue < 0.05:
            set_idx_string = [train_X.index.get_loc(feature) for feature in feature_list]
            filtered_minimal_set_list.append('; '.join(str(v) for v in set_idx_string))
            filtered_minimal_set_rmse_list.append(current_rmse)
        # compare with random features 
        need_for_random_comparision, random_comparison_tuple = check_for_random_comparison(len(feature_list), current_rmse, need_for_random_comparision=need_for_random_comparision, random_comparison_tuple=random_comparison_tuple)
        
        # loop for more disjoint minimal set
        left_over_features = train_X.index.difference(feature_list)
        complementary_features = left_over_features
        while (len(left_over_features) > 0):
        # while False:
            complementary_features = left_over_features
            current_train_X = train_X.loc[complementary_features]
            current_test_X = test_X.loc[complementary_features]
            if (regr_type == 'rf'):
                current_regr = RandomForestRegressor(random_state=42, n_jobs=1, max_features='sqrt' )
            else:
                current_regr = linear_model.Ridge()
            current_regr.fit(current_train_X.T, y_train)
            current_error = np.square(current_regr.predict(current_train_X.T)-y_train)
            if (regr_type == 'rf'):
                current_feature_importance = current_regr.feature_importances_
            else:
                current_feature_importance = np.abs(current_regr.coef_) / np.sum(np.abs(current_regr.coef_))
            s, p = stats.ttest_rel(base_error, current_error)
            current_score = current_regr.score(current_train_X.T, y_train)
            if (current_score < 0):
                continue_flag = False
                complementary_rmse = np.iinfo(np.int16).max
                break
            if (s < 0) and (p < 0.05):
                continue_flag = False
                complementary_rmse = np.iinfo(np.int16).max
                break
            else:
                continue_flag = True
                complementary_rmse = mean_squared_error(
                    current_regr.predict(test_X.loc[complementary_features].T),
                    y_test, squared=False
                )
                complementary_predict = current_regr.predict(test_X.loc[complementary_features].T)
            while (continue_flag):
                half_feature_num = int(len(complementary_features)/2)
                top_half_features = complementary_features[np.argsort(current_feature_importance)[-1*half_feature_num:]]
                current_train_X = train_X.loc[top_half_features]
                current_test_X = test_X.loc[top_half_features]
                if (regr_type == 'rf'):
                    current_regr = RandomForestRegressor(random_state=42, n_jobs=1, max_features='sqrt' )
                else:
                    current_regr = linear_model.Ridge()
                current_regr.fit(current_train_X.T, y_train)
                current_error = np.square(current_regr.predict(current_train_X.T)-y_train)
                s, p = stats.ttest_rel(base_error, current_error)
                current_score = current_regr.score(current_train_X.T, y_train)
                if (current_score < 0):
                    continue_flag = False
                    # print('bad...')
                    break
                if (s < 0) and (p < 0.05):
                    continue_flag = False
                    # print('bad...')
                    break
                complementary_features = top_half_features
                complementary_rmse = mean_squared_error(
                    current_regr.predict(current_test_X.T),
                    y_test, squared=False
                )
                complementary_predict = current_regr.predict(current_test_X.T)
                if (regr_type == 'rf'):
                    current_feature_importance = current_regr.feature_importances_
                else:
                    current_feature_importance = np.abs(current_regr.coef_) / np.sum(np.abs(current_regr.coef_))

                if (len(complementary_features) < 4):
                    continue_flag = False
                    break
            ensemble_predict_list.append(complementary_predict)
            complementary_features_index = [train_X.index.get_loc(feature) for feature in complementary_features]
            complementary_features_rmse_list.append(complementary_rmse)
            complementary_features_idx_list.append('; '.join(str(v) for v in complementary_features_index))
            complementary_features_length_list.append(len(complementary_features_index))
            left_over_features = left_over_features.difference(complementary_features)
            # print(complementary_features_index)
            # print(current_score)
            # print(s,p)

            complementary_squared_error = np.square(complementary_predict-y_test)
            # compare with repeating values
            ref_comp_ttest = stats.ttest_rel(complementary_squared_error, ref_squared_error)
            repeating_value_comp_diff.append(ref_comp_ttest.statistic)
            repeating_value_comp_pval.append(ref_comp_ttest.pvalue)
            if ref_comp_ttest.statistic < 0 and ref_comp_ttest.pvalue < 0.05:
                set_idx_string = [train_X.index.get_loc(feature) for feature in complementary_features]
                filtered_minimal_set_list.append('; '.join(str(v) for v in set_idx_string))
                filtered_minimal_set_rmse_list.append(complementary_rmse)
            
            # compare with random features 
            need_for_random_comparision, random_comparison_tuple = check_for_random_comparison(len(complementary_features), complementary_rmse, need_for_random_comparision=need_for_random_comparision, random_comparison_tuple=random_comparison_tuple)

        # else: complementary_rmse = np.iinfo(np.int16).max
        if (len(complementary_features_rmse_list) < 1):
            complementary_features_rmse_list.append(np.iinfo(np.int16).max)
        feature_list_index = [train_X.index.get_loc(feature) for feature in feature_list]
        # complementary_features_index = [train_X.index.get_loc(feature) for feature in complementary_features]

        ensemble_predict_list = np.array(ensemble_predict_list)
        ensemble_predict = ensemble_predict_list.mean(axis=0)

        return np.array([
            len(feature_list),
            ', '.join(str(v) for v in complementary_features_length_list),
            current_rmse,
            ', '.join(str(v) for v in complementary_features_rmse_list),
            '; '.join(str(v) for v in feature_list_index),
            ', '.join(str(v) for v in complementary_features_idx_list),
            mean_squared_error(ensemble_predict, y_test, squared=False),
            '|'.join(str(v) for v in filtered_minimal_set_list),
            '|'.join(str(v) for v in filtered_minimal_set_rmse_list),
            '|'.join(str(v) for v in repeating_value_comp_diff),
            '|'.join(str(v) for v in repeating_value_comp_pval),
            random_comparison_tuple[0],
            random_comparison_tuple[1]
            ])


    def efron_ensemble_process_rf(self, index):
        return self.efron_ensemble_process(index, 'rf')
    def efron_ensemble_process_linear(self, index):
        return self.efron_ensemble_process(index, 'linear')

    def efron_ensemble_process(self, index, regr_type):
        train_X, test_X, y_train, y_test, tf_list = self.get_train_test_sets(index)
        if (regr_type == 'rf'):
            regr = RandomForestRegressor(random_state=42, n_jobs=1, max_features='sqrt' )
        else:
            regr = linear_model.Ridge()
        regr.fit(train_X.T, y_train)
        base_error = np.square(regr.predict(train_X.T)-y_train)
        feature_list = train_X.index
        if (regr_type == 'rf'):
            current_feature_importance = regr.feature_importances_
        else:
            current_feature_importance = np.abs(regr.coef_) / np.sum(np.abs(regr.coef_))
        current_predict = regr.predict(test_X.T)
        continue_flag = True
        while(continue_flag):
            half_feature_num = int(len(feature_list)/2)
            top_half_features = feature_list[np.argsort(current_feature_importance)[-1*half_feature_num:]]
            current_train_X = train_X.loc[top_half_features]
            current_test_X = test_X.loc[top_half_features]
            if (regr_type == 'rf'):
                current_regr = RandomForestRegressor(random_state=42, n_jobs=1, max_features='sqrt' )
            else:
                current_regr = linear_model.Ridge()
            current_regr.fit(current_train_X.T, y_train)
            current_error = np.square(current_regr.predict(current_train_X.T)-y_train)
            s, p = stats.ttest_rel(base_error, current_error)
            current_score = current_regr.score(current_train_X.T, y_train)
            if (current_score < 0):
                continue_flag = False
                break
            if (s < 0) and (p < 0.05):
                continue_flag = False
                break
            feature_list = top_half_features
            current_predict = current_regr.predict(current_test_X.T)
            if (regr_type == 'rf'):
                current_feature_importance = current_regr.feature_importances_
            else:
                current_feature_importance = np.abs(current_regr.coef_) / np.sum(np.abs(current_regr.coef_))

            if (len(feature_list) < 4):
                continue_flag = False
                break

        ensemble_predict_list = [current_predict]
        # complementary_features_rmse_list = []
        # complementary_features_length_list = []
        # complementary_features_idx_list = []
        left_over_features = train_X.index.difference(feature_list)
        complementary_features = left_over_features
        while (len(left_over_features) > 0):
            complementary_features = left_over_features
            current_train_X = train_X.loc[complementary_features]
            current_test_X = test_X.loc[complementary_features]
            if (regr_type == 'rf'):
                current_regr = RandomForestRegressor(random_state=42, n_jobs=1, max_features='sqrt' )
            else:
                current_regr = linear_model.Ridge()
            current_regr.fit(current_train_X.T, y_train)
            current_error = np.square(current_regr.predict(current_train_X.T)-y_train)
            if (regr_type == 'rf'):
                current_feature_importance = current_regr.feature_importances_
            else:
                current_feature_importance = np.abs(current_regr.coef_) / np.sum(np.abs(current_regr.coef_))
            s, p = stats.ttest_rel(base_error, current_error)
            current_score = current_regr.score(current_train_X.T, y_train)
            if (current_score < 0):
                continue_flag = False
                complementary_rmse = np.iinfo(np.int16).max
                break
            if (s < 0) and (p < 0.05):
                continue_flag = False
                complementary_rmse = np.iinfo(np.int16).max
                break
            else:
                continue_flag = True
                complementary_predict = current_regr.predict(test_X.loc[complementary_features].T)
            while (continue_flag):
                half_feature_num = int(len(complementary_features)/2)
                top_half_features = complementary_features[np.argsort(current_feature_importance)[-1*half_feature_num:]]
                current_train_X = train_X.loc[top_half_features]
                current_test_X = test_X.loc[top_half_features]
                if (regr_type == 'rf'):
                    current_regr = RandomForestRegressor(random_state=42, n_jobs=1, max_features='sqrt' )
                else:
                    current_regr = linear_model.Ridge()
                current_regr.fit(current_train_X.T, y_train)
                current_error = np.square(current_regr.predict(current_train_X.T)-y_train)
                s, p = stats.ttest_rel(base_error, current_error)
                current_score = current_regr.score(current_test_X.T, y_test)
                if (current_score < 0):
                    continue_flag = False
                    # print('bad...')
                    break
                if (s < 0) and (p < 0.05):
                    continue_flag = False
                    # print('bad...')
                    break
                complementary_features = top_half_features
                complementary_predict = current_regr.predict(current_test_X.T)
                if (regr_type == 'rf'):
                    current_feature_importance = current_regr.feature_importances_
                else:
                    current_feature_importance = np.abs(current_regr.coef_) / np.sum(np.abs(current_regr.coef_))

                if (len(complementary_features) < 4):
                    continue_flag = False
                    break
            ensemble_predict_list.append(complementary_predict)
            # complementary_features_index = [train_X.index.get_loc(feature) for feature in complementary_features]
            # complementary_features_rmse_list.append(complementary_rmse)
            # complementary_features_idx_list.append('; '.join(str(v) for v in complementary_features_index))
            # complementary_features_length_list.append(len(complementary_features_index))
            left_over_features = left_over_features.difference(complementary_features)
            # print(complementary_features_index)
            # print(current_score)
            # print(s,p)

        # if (len(complementary_features_rmse_list) < 1):
        #     complementary_features_rmse_list.append(np.iinfo(np.int16).max)
        # feature_list_index = [train_X.index.get_loc(feature) for feature in feature_list]

        ensemble_predict_list = np.array(ensemble_predict_list)
        ensemble_predict = ensemble_predict_list.mean(axis=0)


        return mean_squared_error(ensemble_predict, y_test, squared=False)

    def efron_process_90th_rf(self, index):
        return(self.efron_process_90th(index, 'rf'))

    def efron_process_90th(self, index, regr_type):
        train_X, test_X, y_train, y_test, tf_list = self.get_train_test_sets(index)
        if (regr_type == 'rf'):
            regr = RandomForestRegressor(random_state=42, n_jobs=1, max_features='sqrt' )
        else:
            regr = linear_model.Ridge()
        regr.fit(train_X.T, y_train)
        base_error = np.square(regr.predict(test_X.T)-y_test)
        feature_list = train_X.index
        if (regr_type == 'rf'):
            current_feature_importance = regr.feature_importances_
        else:
            current_feature_importance = np.abs(regr.coef_) / np.sum(np.abs(regr.coef_))
        current_rmse = mean_squared_error(
            regr.predict(test_X.T),
            y_test, squared=False
        )
        continue_flag = True
        while(continue_flag):
            half_feature_num = int(len(feature_list)*0.9)
            top_half_features = feature_list[np.argsort(current_feature_importance)[-1*half_feature_num:]]
            current_train_X = train_X.loc[top_half_features]
            current_test_X = test_X.loc[top_half_features]
            if (regr_type == 'rf'):
                current_regr = RandomForestRegressor(random_state=42, n_jobs=1, max_features='sqrt' )
            else:
                current_regr = linear_model.Ridge()
            current_regr.fit(current_train_X.T, y_train)
            current_error = np.square(current_regr.predict(current_test_X.T)-y_test)
            s, p = stats.ttest_rel(base_error, current_error)
            if (s < 0) and (p < 0.05):
                continue_flag = False
                break
            if (half_feature_num < 4):
                continue_flag = False
                break
            feature_list = top_half_features
            current_rmse = mean_squared_error(
                current_regr.predict(current_test_X.T),
                y_test, squared=False
            )
            if (regr_type == 'rf'):
                current_feature_importance = current_regr.feature_importances_
            else:
                current_feature_importance = np.abs(current_regr.coef_) / np.sum(np.abs(current_regr.coef_))

        complementary_features = train_X.index.difference(feature_list)
        if (len(complementary_features) > 0):
            if (regr_type == 'rf'):
                complementary_regr = RandomForestRegressor(random_state=42, n_jobs=1, max_features='sqrt' )
            else:
                complementary_regr = linear_model.Ridge()
            complementary_regr.fit(train_X.loc[complementary_features].T, y_train)
            complementary_rmse = mean_squared_error(
                complementary_regr.predict(test_X.loc[complementary_features].T),
                y_test, squared=False
            )
        else: complementary_rmse = np.sqrt(np.mean(base_error))
        return np.array([len(feature_list), current_rmse, complementary_rmse])

    def dynamic_efron_rf(self, index):
        return(self.dynamic_efron(index, 'rf'))
    def dynamic_efron_linear(self, index):
        return(self.dynamic_efron(index, 'linear'))

    def dynamic_efron(self, index, regr_type):
        train_X, test_X, y_train, y_test, tf_list = self.get_train_test_sets(index)
        if (regr_type == 'rf'):
            regr = RandomForestRegressor(random_state=42, n_jobs=1, max_features='sqrt' )
        else:
            regr = linear_model.Ridge()
        regr.fit(train_X.T, y_train)
        base_error = np.square(regr.predict(test_X.T)-y_test)
        feature_list = train_X.index
        if (regr_type == 'rf'):
            base_feature_importance = regr.feature_importances_
        else:
            base_feature_importance = np.abs(regr.coef_) / np.sum(np.abs(regr.coef_))

        current_rmse = mean_squared_error(
            regr.predict(test_X.T),
            y_test, squared=False
        )
        current_feature_importance = base_feature_importance
        base_feature_importance_presum = np.cumsum(base_feature_importance)
        continue_flag = True
        top_feature_percent = 0.99
        current_feature_list = feature_list
        while(continue_flag):
            top_feature_num = np.argmax(base_feature_importance_presum>=top_feature_percent) + 1
            top_features = feature_list[np.argsort(base_feature_importance)[-1*top_feature_num:]]
            current_train_X = train_X.loc[top_features]
            current_test_X = test_X.loc[top_features]
            if (regr_type == 'rf'):
                current_regr = RandomForestRegressor(random_state=42, n_jobs=1, max_features='sqrt' )
            else:
                current_regr = linear_model.Ridge()
            current_regr.fit(current_train_X.T, y_train)
            current_error = np.square(current_regr.predict(current_test_X.T)-y_test)
            s, p = stats.ttest_rel(base_error, current_error)
            if (s < 0) and (p < 0.05):
                continue_flag = False
                break
            current_feature_list = top_features
            current_rmse = mean_squared_error(
                current_regr.predict(current_test_X.T),
                y_test, squared=False
            )
            top_feature_percent -= 0.01
            if (len(current_feature_list) < 4):
                continue_flag = False
                break

        top_feature_percent += 0.01
        top_feature_num = np.argmax(base_feature_importance_presum>=top_feature_percent) + 1
        top_features = feature_list[np.argsort(base_feature_importance)[-1*top_feature_num:]]
        current_feature_index = np.argsort(base_feature_importance)[-1*top_feature_num:]

        complementary_features = train_X.index.difference(current_feature_list)
        if (len(complementary_features) > 0):
            if (regr_type == 'rf'):
                complementary_regr = RandomForestRegressor(random_state=42, n_jobs=1, max_features='sqrt' )
            else:
                complementary_regr = linear_model.Ridge()
            complementary_regr.fit(train_X.loc[complementary_features].T, y_train)
            complementary_rmse = mean_squared_error(
                complementary_regr.predict(test_X.loc[complementary_features].T),
                y_test, squared=False
            )
        else: complementary_rmse = np.sqrt(np.mean(base_error))

        return np.array([len(current_feature_list), current_rmse, complementary_rmse, '; '.join(str(v) for v in current_feature_index)])

    def rf_top_tf_same_count_as_gs(self, index):
        train_X, test_X, y_train, y_test, tf_list = self.get_train_test_sets(index)
        rf_regr = RandomForestRegressor(random_state=42, n_jobs=1, max_features='sqrt' )
        rf_regr.fit(train_X.T, y_train)
        rf_top_feature_num = len(tf_list)
        top_rf_tf_list = np.flip(rf_regr.feature_names_in_[np.argsort(rf_regr.feature_importances_)[-rf_top_feature_num:]])
        top_rf_regr = RandomForestRegressor(random_state=45, n_jobs=1, max_features='sqrt' )
        top_rf_regr.fit(train_X.loc[top_rf_tf_list].T, y_train)
        return np.array([
            top_rf_regr.score(test_X.loc[top_rf_tf_list].T, y_test),
            mean_squared_error(top_rf_regr.predict(test_X.loc[top_rf_tf_list].T), y_test, squared=False)
        ])

    def rf_top_10(self, index):
        train_X, test_X, y_train, y_test, tf_list = self.get_train_test_sets(index)
        rf_regr = RandomForestRegressor(random_state=42, n_jobs=1, max_features='sqrt' )
        rf_regr.fit(train_X.T, y_train)
        rf_top_feature_num = 10
        top_rf_tf_list = np.flip(rf_regr.feature_names_in_[np.argsort(rf_regr.feature_importances_)[-rf_top_feature_num:]])
        top_rf_regr = RandomForestRegressor(random_state=45, n_jobs=1, max_features='sqrt' )
        top_rf_regr.fit(train_X.loc[top_rf_tf_list].T, y_train)
        return np.array([
            top_rf_regr.score(test_X.loc[top_rf_tf_list].T, y_test),
            mean_squared_error(top_rf_regr.predict(test_X.loc[top_rf_tf_list].T), y_test, squared=False)
        ])

    def full_comp_new(self, index, debug=False):
        train_X, test_X, y_train, y_test, tf_list = self.get_train_test_sets(index)
        if (debug):
            print(train_X.shape)
            print(test_X.shape)
            print(y_train.shape)
            print(y_test.shape)
        rf_regr = RandomForestRegressor(random_state=42, n_jobs=1, max_features='sqrt' )
        gs_rf_regr = RandomForestRegressor(random_state=43, n_jobs=1, max_features='sqrt' )
        linear_regr = linear_model.Ridge()
        gs_linear_regr = linear_model.Ridge()
        pca = PCA(n_components=3)
        pca.fit(train_X.T)
        pca_train_X = pca.transform(train_X.T)
        pca_test_X = pca.transform(test_X.T)
        pca_rf_regr = RandomForestRegressor(random_state=45, n_jobs=1, max_features='sqrt' )
        rf_regr.fit(train_X.T, y_train)
        gs_rf_regr.fit(train_X.loc[tf_list].T, y_train)
        linear_regr.fit(train_X.T, y_train)
        gs_linear_regr.fit(train_X.loc[tf_list].T, y_train)
        pca_rf_regr.fit(pca_train_X, y_train)

        rf_presum_feature_importance = np.cumsum(np.flip(np.sort(rf_regr.feature_importances_)))
        linear_feature_importance = np.abs(linear_regr.coef_) / np.sum(np.abs(linear_regr.coef_))
        linear_presum_feature_importance = np.cumsum(np.flip(np.sort(linear_feature_importance)))

        rf_top_feature_num = np.argmax(rf_presum_feature_importance>0.95) + 1
        rf_top_feature_num = 20
        linear_top_feature_num = np.argmax(linear_presum_feature_importance>0.95) + 1
        linear_top_feature_num = 20

        top_rf_tf_list = np.flip(rf_regr.feature_names_in_[np.argsort(rf_regr.feature_importances_)[-rf_top_feature_num:]])
        top_linear_tf_list = np.flip(rf_regr.feature_names_in_[np.argsort(np.abs(linear_regr.coef_))[-linear_top_feature_num:]])

        new_rf_regr = RandomForestRegressor(random_state=44, n_jobs=1, max_features='sqrt' )
        new_linear_regr = linear_model.Ridge()
        new_rf_regr.fit(train_X.loc[top_linear_tf_list].T, y_train)
        new_linear_regr.fit(train_X.loc[top_rf_tf_list].T, y_train)

        top_rf_regr = RandomForestRegressor(random_state=45, n_jobs=1, max_features='sqrt' )
        top_linear_regr = linear_model.Ridge()
        top_rf_regr.fit(train_X.loc[top_rf_tf_list].T, y_train)
        top_linear_regr.fit(train_X.loc[top_linear_tf_list].T, y_train)

        top_rf_feature_gs_overlap_num = len(set(top_rf_tf_list).intersection(set(tf_list)))
        top_linear_feature_gs_overlap_num = len(set(top_linear_tf_list).intersection(set(tf_list)))
        top_linear_rf_feature_overlap_num = len(set(top_linear_tf_list).intersection(set(top_rf_tf_list)))

        return np.array([
            rf_regr.score(test_X.T, y_test),
            linear_regr.score(test_X.T, y_test),
            gs_rf_regr.score(test_X.loc[tf_list].T, y_test),
            gs_linear_regr.score(test_X.loc[tf_list].T, y_test),
            new_rf_regr.score(test_X.loc[top_linear_tf_list].T, y_test),
            new_linear_regr.score(test_X.loc[top_rf_tf_list].T, y_test),
            mean_squared_error(rf_regr.predict(test_X.T), y_test, squared=False),
            mean_squared_error(linear_regr.predict(test_X.T), y_test, squared=False),
            mean_squared_error(gs_rf_regr.predict(test_X.loc[tf_list].T), y_test, squared=False),
            mean_squared_error(gs_linear_regr.predict(test_X.loc[tf_list].T), y_test, squared=False),
            mean_squared_error(new_rf_regr.predict(test_X.loc[top_linear_tf_list].T), y_test, squared=False),
            mean_squared_error(new_linear_regr.predict(test_X.loc[top_rf_tf_list].T), y_test, squared=False),
            top_rf_regr.score(test_X.loc[top_rf_tf_list].T, y_test),
            top_linear_regr.score(test_X.loc[top_linear_tf_list].T, y_test),
            mean_squared_error(top_rf_regr.predict(test_X.loc[top_rf_tf_list].T), y_test, squared=False),
            mean_squared_error(top_linear_regr.predict(test_X.loc[top_linear_tf_list].T), y_test, squared=False),
            rf_top_feature_num,
            linear_top_feature_num,
            top_rf_feature_gs_overlap_num,
            top_linear_feature_gs_overlap_num,
            top_linear_rf_feature_overlap_num,
            len(tf_list),
            y_test.var(),
            y_test.std(),
            pca_rf_regr.score(pca_test_X, y_test),
            mean_squared_error(pca_rf_regr.predict(pca_test_X), y_test, squared=False)
        ])


    def all_linear_comp(self, index):
        target = self.target_gene_list[index]
        tf_list = self.network_df.loc[target].tf_list
        tf_list = tf_list.split('; ')
        tf_list = list(self.available_tfs.intersection(tf_list))
        y_train = self.train_target.loc[target]
        y_test = self.test_target.loc[target]
        X_train = self.train_source.T
        X_test = self.test_source.T
        X_train_gs = self.train_source.loc[tf_list].T
        X_test_gs = self.test_source.loc[tf_list].T

        regr_models = [
            linear_model.LinearRegression(),
            linear_model.Ridge(),
            linear_model.Lasso(),
            linear_model.ElasticNet(),
            linear_model.BayesianRidge(),
            linear_model.QuantileRegressor(),
            Pipeline([('quadratic', PolynomialFeatures(2)), ('linear', linear_model.LinearRegression())])
        ]

        res = []
        for regr in regr_models:
            regr.fit(X_train, y_train)
            res.append(regr.score(X_test, y_test))

        for regr in regr_models:
            regr.fit(X_train_gs, y_train)
            res.append(regr.score(X_test_gs, y_test))

        return np.array(res)

    def full_comp_runner(self, target_gene_list, output_path, cpu_cores=1, skip_comp=False):
        out_df = pd.DataFrame(index=target_gene_list)
        iter_length = len(target_gene_list)
        if (not skip_comp):
            print('Comparing different regression approaches... ...')
            start_time = time.time()
            print('Step 1 of 3:')
            with Pool(cpu_cores) as p:
                r = list(tqdm(p.imap(self.full_comp_new, range(iter_length)), total=iter_length))
            r = np.array(r)
            out_df = pd.DataFrame(index=target_gene_list)
            out_df['rf_score'] = r[:, 0]
            out_df['linear_score'] = r[:, 1]
            out_df['gs_rf_score'] = r[:, 2]
            out_df['gs_linear_score'] = r[:, 3]
            out_df['rf_with_linear_top_features_score'] = r[:, 4]
            out_df['linear_with_rf_top_features_score'] = r[:, 5]
            out_df['rf_rmse'] = r[:, 6]
            out_df['linear_rmse'] = r[:, 7]
            out_df['gs_rf_rmse'] = r[:, 8]
            out_df['gs_linear_rmse'] = r[:, 9]
            out_df['rf_with_linear_top_features_rmse'] = r[:, 10]
            out_df['linear_with_rf_top_features_rmse'] = r[:, 11]
            out_df['rf_with_top_features_score'] = r[:, 12]
            out_df['linear_with_top_features_score'] = r[:, 13]
            out_df['rf_with_top_features_rmse'] = r[:, 14]
            out_df['linear_with_top_features_rmse'] = r[:, 15]
            out_df['rf_top_feature_num'] = r[:, 16]
            out_df['linear_top_feature_num'] = r[:, 17]
            out_df['rf_top_features_gs_overlap'] = r[:, 18]
            out_df['linear_top_features_gs_overlap'] = r[:, 19]
            out_df['rf_linear_top_features_overlap'] = r[:, 20]
            out_df['gs_edge_num'] = r[:, 21]
            out_df['test_var'] = r[:, 22]
            out_df['test_std'] = r[:, 23]
            out_df['pca_rf_score'] = r[:, 24]
            out_df['pca_rf_rmse'] = r[:, 25]

            print('Step 2 of 3:')
            with Pool(cpu_cores) as p:
                r = list(tqdm(p.imap(self.rf_top_tf_same_count_as_gs, range(iter_length)), total=iter_length))
            r = np.array(r)
            out_df['rf_top_tf_same_count_as_gs_score'] = r[:, 0]
            out_df['rf_top_tf_same_count_as_gs_rmse'] = r[:, 1]

            print('Step 3 of 3:')
            with Pool(cpu_cores) as p:
                r = list(tqdm(p.imap(self.rf_top_10, range(iter_length)), total=iter_length))
            r = np.array(r)
            out_df['rf_top10_score'] = r[:, 0].astype('float64')
            out_df['rf_top10_rmse'] = r[:, 1].astype('float64')

            end_time = time.time()
            print('Finished comparing all approaches, time elapsed: {} seconds'.format(end_time-start_time))

        start_time = time.time()
        print('Calculating minimal set and disjoint sets... ...')
        print('Step 1 of 1:')
        with Pool(cpu_cores) as p:
            r = list(tqdm(p.imap(self.efron_process_rf_training, range(iter_length)), total=iter_length))
        efron_r = np.array(r)
        out_df['rf_efron_feature_num'] = efron_r[:, 0].astype('float64')
        out_df['rf_efron_complementary_feature_num_list'] = efron_r[:, 1]
        out_df['rf_efron_rmse'] = efron_r[:, 2].astype('float64')
        out_df['rf_efron_complementary_rmse_list'] = efron_r[:, 3]
        out_df['rf_efron_features'] = efron_r[:, 4]
        out_df['rf_efron_complementary_features_list'] = efron_r[:, 5]
        out_df['rf_efron_ensemble_rmse'] = efron_r[:, 6]
        out_df['rf_efron_filtered_minimal_sets'] = efron_r[:, 7]
        out_df['rf_efron_filtered_minimal_sets_rmse'] = efron_r[:, 8]
        out_df['rf_efron_repeating_value_comp_diff'] = efron_r[:, 9]
        out_df['rf_efron_repeating_value_comp_efron'] = efron_r[:, 10]
        out_df['rf_efron_comp_rmse'] = efron_r[:, 11]
        out_df['rf_efron_random_rmse'] = efron_r[:, 12]

        rf_efron_overlap_count = []
        tf_list_df = pd.DataFrame(index=self.tf_list)
        for target_gene in out_df.index:
            gs_tf_list = self.network_df.loc[target_gene].tf_list
            gs_tf_set = set(gs_tf_list.split('; '))
            gs_tf_set = self.available_tfs.intersection(gs_tf_set)
            if target_gene in gs_tf_set: gs_tf_set.remove(target_gene)
            efron_tf_list = out_df.loc[target_gene]['rf_efron_features']
            efron_tf_list = efron_tf_list.split('; ')
            efron_tf_list = [int(i) for i in efron_tf_list]
            efron_tf_list = tf_list_df.iloc[efron_tf_list].index
            efron_tf_set = set(efron_tf_list)
            rf_efron_overlap_count.append(len(efron_tf_set.intersection(gs_tf_set)))
        out_df['rf_efron_overlap_count'] = rf_efron_overlap_count
        end_time = time.time()
        print('Finished calculating minimal set and disjoint sets, time elapsed: {} seconds'.format(end_time- start_time))

        out_df.to_csv(output_path, compression='gzip')


