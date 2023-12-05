# candidate = AT5G44610

import pandas as pd
import numpy as np
from tqdm import tqdm

import mp_run


from scipy import stats

from multiprocessing import Pool, cpu_count



# regex for number extraction from string
number_pattern =  r'(-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)'


train_source_df = pd.read_csv('../data/GSE97500/train_source_tpm.csv', index_col=0)
train_target_df = pd.read_csv('../data/GSE97500/train_target_tpm.csv', index_col=0)

test_source_df = pd.read_csv('../data/GSE97500/test_source_tpm.csv', index_col=0)
test_target_df = pd.read_csv('../data/GSE97500/test_target_tpm.csv', index_col=0)


train_source_df = train_source_df.apply(stats.zscore, axis=0)
train_target_df = train_target_df.apply(stats.zscore, axis=0)

test_source_df = test_source_df.apply(stats.zscore, axis=0)
test_target_df = test_target_df.apply(stats.zscore, axis=0)


regulator_set = set()
tf_list_df = pd.read_csv('../data/arabidopsis_tf_list.tsv.gz', sep='\t', compression='gzip', index_col=0)
for name in tf_list_df['Gene Names']:
    name_splits = name.split(' ')
    for i in name_splits:
        if i.upper() in train_source_df.index:
            regulator_set.add(i.upper())

network_df = pd.read_csv('../data/arabidopsis_network_connectf.csv', index_col=0)
target_set = set(network_df.index)

for tf_list_string in network_df['tf_list'].values:
    tf_list = tf_list_string.split('; ')
    for tf in tf_list: regulator_set.add(tf)

regulator_set = regulator_set.intersection(set(train_source_df.index))
target_set = target_set.intersection(set(train_source_df.index))
all_gene_set = regulator_set.union(target_set)



train_source = train_source_df.loc[list(all_gene_set)].apply(stats.zscore, axis=0)
train_target = train_target_df.loc[list(all_gene_set)].apply(stats.zscore, axis=0)

test_source = test_source_df.loc[list(all_gene_set)].apply(stats.zscore, axis=0)
test_target = test_target_df.loc[list(all_gene_set)].apply(stats.zscore, axis=0)


print(train_source.shape)
print(test_source.shape)


target_df = pd.concat([train_target, test_target], axis=1)
source_df = pd.concat([train_source, test_source], axis=1)


target_gene_list = list(target_set)
target_exp = target_df
X = source_df.loc[list(regulator_set)]
tf_list = list(regulator_set)

new_test_target = test_target.loc[target_gene_list]
new_test_target = new_test_target.loc[new_test_target.std(axis=1) > 0.5]
target_gene_list = new_test_target.index

mp_calc = mp_run.MpCalc(target_gene_list, target_exp, X, network_df, train_source.loc[tf_list], train_target, test_source.loc[tf_list], test_target)

iter_length = len(target_gene_list)
with Pool(cpu_count()) as p:
    r = list(tqdm(p.imap(mp_calc.full_comp_new, range(iter_length)), total=iter_length))

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

iter_length = len(target_gene_list)
with Pool(cpu_count()) as p:
    r = list(tqdm(p.imap(mp_calc.efron_process_rf, range(iter_length)), total=iter_length))

efron_r = np.array(r)
out_df['rf_efron_feature_num'] = efron_r[:, 0].astype('float64')
out_df['rf_efron_complementary_feature_num_list'] = efron_r[:, 1]
out_df['rf_efron_rmse'] = efron_r[:, 2].astype('float64')
out_df['rf_efron_complementary_rmse_list'] = efron_r[:, 3]
out_df['rf_efron_features'] = efron_r[:, 4]
out_df['rf_efron_complementary_features_list'] = efron_r[:, 5]


tf_list_df = pd.DataFrame(index=tf_list)
tf_list_df.to_csv('../output/network_model/arabidopsis_tf.csv', header=False)
out_df.to_csv('../output/network_model/arabidopsis_all_tf_high_var_target_new.csv.gz', compression='gzip')

