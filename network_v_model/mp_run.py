from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import numpy as np
from xgboost import XGBRegressor
from xgboost import XGBRFRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import explained_variance_score, mean_squared_error

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from scipy import stats

import warnings
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
        self.available_tfs = set(X.index)

    def get_train_test_sets(self, index):
        target = self.target_gene_list[index]
        tf_list = self.network_df.loc[target].tf_list
        tf_list = tf_list.split('; ')
        tf_list = list(self.available_tfs.intersection(tf_list))
        if target in tf_list: tf_list.remove(target)
        if target in self.train_source.index:
            train_X = self.train_source.drop([target])
            test_X = self.test_source.drop([target])
        else:
            train_X = self.train_source
            test_X = self.test_source
        y_train = self.train_target.loc[target]
        y_test = self.test_target.loc[target]
        return train_X, test_X, y_train, y_test, tf_list
    
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

    
    def full_comp_new(self, index):
        train_X, test_X, y_train, y_test, tf_list = self.get_train_test_sets(index)
        rf_regr = RandomForestRegressor(random_state=42, n_jobs=1, max_features='sqrt' )
        gs_rf_regr = RandomForestRegressor(random_state=43, n_jobs=1, max_features='sqrt' )
        linear_regr = linear_model.Ridge()
        gs_linear_regr = linear_model.Ridge()
        rf_regr.fit(train_X.T, y_train)
        gs_rf_regr.fit(train_X.loc[tf_list].T, y_train)
        linear_regr.fit(train_X.T, y_train)
        gs_linear_regr.fit(train_X.loc[tf_list].T, y_train)
        
        rf_presum_feature_importance = np.cumsum(np.flip(np.sort(rf_regr.feature_importances_)))
        linear_feature_importance = np.abs(linear_regr.coef_) / np.sum(np.abs(linear_regr.coef_))
        linear_presum_feature_importance = np.cumsum(np.flip(np.sort(linear_feature_importance)))

        rf_top_feature_num = np.argmax(np.cumsum(rf_presum_feature_importance)>0.95) + 1
        linear_top_feature_num = np.argmax(np.cumsum(linear_presum_feature_importance)>0.95) + 1
        
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
            y_test.std()
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
        

