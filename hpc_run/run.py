import pandas as pd
import numpy as np
from tqdm import tqdm

perturbation_factor = 3

from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

ts_df = pd.read_csv('expression.tsv', sep='\t', index_col=0)

deg_df = pd.read_csv('92_DEG_Clusters.csv', index_col=0)
deg_genes = deg_df.index

target_genes = pd.Series(list(set(deg_genes).intersection(set(ts_df.index))))

meta_df = pd.read_csv('meta_data.tsv', sep='\t')
ts_exp_index = meta_df[meta_df['isTs']]

ts_exp_index_target =  ts_exp_index[ts_exp_index['is1stLast'] != 'f'].condName
ts_exp_index_source =  ts_exp_index[ts_exp_index['is1stLast'] != 'f'].prevCol
regulator_gene_index = ts_df.index

from sklearn.ensemble import RandomForestRegressor

regr = RandomForestRegressor(random_state=0)

ts_train_y_list = ts_df[ts_exp_index_target]

result_list = []
result_measure_list = []

for target_gene in tqdm(target_genes):
    train_gene_index = regulator_gene_index[regulator_gene_index != target_gene]
    ts_train_X = ts_df[ts_exp_index_source].T[train_gene_index]
    
    ts_train_y = ts_train_y_list.loc[target_gene]
    importance_matrix = np.zeros((len(train_gene_index),500))
    for i in range(500):
        regr = RandomForestRegressor(random_state=i, warm_start=True, n_estimators=300, n_jobs=20)
        regr = regr.fit(ts_train_X, ts_train_y)
        importance_matrix[:,i] = regr.feature_importances_

    importance_df = pd.DataFrame(index=train_gene_index, data=importance_matrix, columns=range(500))

    mean_importance = importance_df.mean(axis=1)
    top_influence_genes = train_gene_index[np.argsort(mean_importance)[-5:]]
    data_mean = ts_df.T[top_influence_genes].mean()
    data_std = ts_df.T[top_influence_genes].std()
    regr = RandomForestRegressor(random_state=42, warm_start=True, n_estimators=100, n_jobs=20)
    ts_train_X = ts_df[ts_exp_index_source].T[top_influence_genes]
    regr = regr.fit(ts_train_X, ts_train_y)

    base_prediction = regr.predict(np.array(data_mean).reshape(1,-1))[0]
    y_std = ts_df.T.std()[target_gene]
    perturbation_list = list(powerset(top_influence_genes))[1:]

    perturbation_result_list = []
    perturbation_list_names = ['; '.join(perturbation_genes) for perturbation_genes in perturbation_list]
    for perturbation_genes in perturbation_list:
        perturbation_input = data_mean.copy()
        for gene in perturbation_genes:
            perturbation_input[gene] += data_std[gene] * perturbation_factor
        perturbation_prediction = regr.predict(np.array(perturbation_input).reshape(1,-1))[0]
        perturbation_measure = (perturbation_prediction - base_prediction)/y_std
        perturbation_result_list.append(perturbation_measure)
    result_list.append(np.array(perturbation_list_names)[np.argsort(perturbation_result_list)[::-1][:5]])
    result_measure_list.append(np.array(perturbation_result_list)[np.argsort(perturbation_result_list)[::-1][:5]])
    

result_measure_list = np.array(result_measure_list)
result_list = np.array(result_list)
out_df = pd.DataFrame()
out_df.index = target_genes
for i in range(5):
    comb_name = 'top_{}_combination'.format(i+1)
    score_name = 'top_{}_score'.format(i+1)
    out_df[comb_name] = result_list[:,i]
    out_df[score_name] = result_measure_list[:,i]

out_df.to_csv('aug27.csv')
