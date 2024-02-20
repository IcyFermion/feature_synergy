## Data and code repository for bipartite gene regulatory network paper

#### Data processing:
Data processing scripts are included in the respective GEO RNA-seq folders under `data` directory. These scripts will format the time series data into z-score normalized traning/testing data. And since we are using sequnce data at prior time points to predict expression at later time points, each dataset are then divided to source and target respectively.

#### Experiments
To carry out all comparisons between different regression models and to calculate disjoint sets of TFs for different species, use the IPython scripts: `network_v_model/*_network_v_model.ipynb`. The scripts also output some basic statistic information about the experiments that were presented in the paper

#### Plotting
The majority of tables and figures used in the paper were generated through `network_v_model/plot_for_paper.ipynb`. Additional scripts `network_v_model/*_batch_comp.ipynb` were used to generate comparisions for batch effects


