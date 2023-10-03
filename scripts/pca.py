# %%
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
import os
from src.auxiliary_fcts_opt import *

from scipy.stats import linregress, norm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from multiprocessing import Pool
from sklearn.cluster import DBSCAN, HDBSCAN
import plotly.express as px

if 'tissue' not in globals():
    tissue = 'Blood'  # Assign a value if tissue has not been defined

with open(f'../exports/pan_mammal_{tissue}.pk', 'rb') as f:
    data_list = pickle.load(f)
# data = data_list[0]
# all_sites = set.union(*[set(data[np.abs(data.obs.slope)> 0.2/data.uns['lifespan']].obs.index) for data in data_list])
# data_list = [data[list(all_sites)] for data in data_list]

results_path = f'../results/{tissue}/pca/'
# Check if the directory exists
if not os.path.exists(results_path):
    # If it doesn't exist, create it
    os.makedirs(results_path)

for data in data_list:
    if data.uns['organism'] == 'Cervus elaphus':
        data.uns['common_name'] = 'Red deer'
    if data.uns['organism'] == 'Cervus canadensis':
        data.uns['common_name'] = 'Wapity elk'

#combine all data.X.T into one array, but keep track of which species each row is from
data = np.concatenate([data.X.T for data in data_list], axis=0)
species = np.concatenate([[data.uns['common_name']] * len(data.X.T) for data in data_list], axis=0)
lifespan = np.concatenate([[data.uns['lifespan']] * len(data.X.T) for data in data_list], axis=0)

#get ages for each row
ages = np.concatenate([data.var['age'].values for data in data_list], axis=0)
female = np.concatenate([data.var['female'].values for data in data_list], axis=0)



from sklearn.decomposition import PCA
pca = PCA(n_components=3)
results = pca.fit(data)
pca_data = pca.transform(data)
pca_data = pd.DataFrame(pca_data, columns=['PC1', 'PC2', 'PC3'])
pca_data['age'] = ages
pca_data['female'] = female
pca_data['common_name'] = species
pca_data['mean_meth'] = data.mean(axis=1)
pca_data['lifespan'] = lifespan


# Compute DBScan clusters
pca_data['cluster'] = 0

max_cluster = 1
min_distance = 1.5
for species in pca_data.common_name.unique():
    sub_data = pca_data[pca_data.common_name == species]
    min_samples = int(len(sub_data)*0.05)
    min_samples = max(min_samples, 5)
    clustering = DBSCAN(eps=min_distance, min_samples=min_samples).fit(sub_data[['PC1', 'PC2']])
    clustering.labels_ = np.where(clustering.labels_==-1, np.nan, clustering.labels_)
    max_cluster += np.nanmax(clustering.labels_) + 1

    pca_data.loc[pca_data.common_name == species, 'cluster'] = (
        clustering.labels_ + max_cluster)

# replace nan with 0
pca_data.cluster = np.nan_to_num(np.array(pca_data.cluster))

# Higlight outliers
pca_data['outlier'] = True
pca_data.loc[pca_data.cluster > 0,'outlier'] = False


fig = sns.scatterplot(pca_data, x='PC1', y='PC2', hue='common_name')
sns.despine()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig(results_path + 'PCA_species.png', bbox_inches='tight')
plt.clf()
fig = sns.scatterplot(pca_data, x='PC1', y='PC2', hue='outlier')
sns.despine()
plt.savefig(results_path + 'PCA_outliers.png')

fig = px.scatter(pca_data, x='PC1', y='PC2', color='common_name', hover_name=pca_data['common_name'])
fig.write_html(results_path + 'PCA_species.html')

fig = px.scatter(pca_data, x='PC1', y='PC2', color='outlier', hover_name=pca_data['common_name'])
fig = px.scatter(pca_data, x='PC1', y='PC2', color='outlier', hover_name=pca_data['age'])
fig.write_html(results_path + 'PCA_outliers.html')

for data in data_list:
    sub_pca = pca_data[pca_data['common_name'] == data.uns['common_name']]
    data.var['PC1'] = sub_pca['PC1'].values
    data.var['PC2'] = sub_pca['PC2'].values
    data.var['cluster'] = sub_pca['cluster'].values
    data.var['outlier'] = sub_pca['outlier'].values

with open(f'../exports/pan_mammal_{tissue}_outliers.pk', 'wb') as f:
    pickle.dump(data_list, f)
# %%