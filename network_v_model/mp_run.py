from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
from xgboost import XGBRegressor
from xgboost import XGBRFRegressor
from sklearn.model_selection import train_test_split, cross_val_score

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore")

class MpCalc:
    def __init__(self, target_gene_list, target_exp, X, network_df, train_source, train_target, test_source, test_target):
        self.target_gene_list = target_gene_list
        self.target_exp = target_exp
        self.X = X
        self.network_df = network_df
        self.train_source = train_source
        self.train_target = train_target
        self.test_source = test_source
        self.test_target = test_target
    
    def calc(self, index):
        target = self.target_gene_list[index]
        y = self.target_exp.loc[target]
        xb_regr = RandomForestRegressor(random_state=42, n_jobs=1)
        scores = cross_val_score(xb_regr, self.X.T, y, cv=5)
        return np.mean(scores)
    
    def all_linear_calc(self, index):
        target = self.target_gene_list[index]
        y = self.target_exp.loc[target]
        regr = LinearRegression()
        scores = cross_val_score(regr, self.X.T, y, cv=5)
        return np.mean(scores)
    
    def network_v_model(self, index):
        target = self.target_gene_list[index]
        y = self.target_exp.loc[target]
        xb_regr = RandomForestRegressor(random_state=42, n_jobs=1)
        # xb_regr = xgb.XGBRegressor(random_state=42, n_jobs=1)
        linear_regr = LinearRegression()
        linear_scores = []
        tf_list = self.network_df.loc[target].tf_list
        tf_list = tf_list.split('; ')
        for tf in tf_list:
            X_tf = self.X.loc[tf]
            scores = cross_val_score(linear_regr, np.array([X_tf]).T, y, cv=5)
            linear_scores.append(np.mean(scores))
        model_cv_scores = cross_val_score(xb_regr, self.X.T, y, cv=5)

        return np.array([np.max(linear_scores), np.mean(linear_scores), np.mean(model_cv_scores)])
    
    def network_all_v_model(self, index):
        target = self.target_gene_list[index]
        y = self.target_exp.loc[target]
        xb_regr = RandomForestRegressor(random_state=42, n_jobs=1)
        # xb_regr = xgb.XGBRegressor(random_state=42, n_jobs=1)
        linear_regr = LinearRegression()
        tf_list = self.network_df.loc[target].tf_list
        tf_list = tf_list.split('; ')
        X_tf = self.X.loc[tf_list]
        linear_scores = cross_val_score(linear_regr, X_tf.T, y, cv=5)
        model_cv_scores = cross_val_score(xb_regr, self.X.T, y, cv=5)

        return np.array([np.mean(linear_scores), np.mean(model_cv_scores)])
    
    def lime_test(self, index):
        target = self.target_gene_list[index]
        tf_list = self.network_df.loc[target].tf_list
        tf_list = tf_list.split('; ')
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
    
    def full_comp(self, index):
        target = self.target_gene_list[index]
        tf_list = self.network_df.loc[target].tf_list
        tf_list = tf_list.split('; ')
        y_train = self.train_target.loc[target]
        y_test = self.test_target.loc[target]
        rf_regr = RandomForestRegressor(random_state=42, n_jobs=1, max_features='sqrt' )
        gs_rf_regr = RandomForestRegressor(random_state=43, n_jobs=1, max_features='sqrt' )
        linear_regr = LinearRegression()
        gs_linear_regr = LinearRegression()
        rf_regr.fit(self.train_source.T, y_train)
        gs_rf_regr.fit(self.train_source.loc[tf_list].T, y_train)
        linear_regr.fit(self.train_source.T, y_train)
        gs_linear_regr.fit(self.train_source.loc[tf_list].T, y_train)
        
        top_rf_tf_list = np.flip(rf_regr.feature_names_in_[np.argsort(rf_regr.feature_importances_)[-10:]])
        top_linear_tf_list = np.flip(rf_regr.feature_names_in_[np.argsort(np.abs(linear_regr.coef_))[-10:]])
        
        new_rf_regr = RandomForestRegressor(random_state=44, n_jobs=1, max_features='sqrt' )
        new_linear_regr = LinearRegression()
        new_rf_regr.fit(self.train_source.loc[top_linear_tf_list].T, y_train)
        new_linear_regr.fit(self.train_source.loc[top_rf_tf_list].T, y_train)
        
        return np.array([
            rf_regr.score(self.test_source.T, y_test), 
            linear_regr.score(self.test_source.T, y_test),
            gs_rf_regr.score(self.test_source.loc[tf_list].T, y_test),
            gs_linear_regr.score(self.test_source.loc[tf_list].T, y_test),
            new_rf_regr.score(self.test_source.loc[top_linear_tf_list].T, y_test),
            new_linear_regr.score(self.test_source.loc[top_rf_tf_list].T, y_test),
        ])
        
        

