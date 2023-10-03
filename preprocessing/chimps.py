import sys
sys.path.append("../")   # fix to import modules from root
from src.general_imports import *
from src.preprocessing_fcts import evaluate_peaks
import GEOparse as geo
import os
import pymc as pm
from sklearn.feature_selection import r_regression

path_prefix = '../'

# import anage data
anage = pd.read_csv(path_prefix + 'data/anage_data.txt', sep='\t', on_bad_lines='skip')

# import mammalian sites
with open("../resources/mammalian_sites.json", "r") as outfile:
    mammalian_sites = json.load(outfile)

animal = 'chimps'
GSE = 'GSE136296'
genus = 'Pan'
species = 'troglodytes'

# path to data
data_path = path_prefix + 'data/GSE136296_eegoguevara.2019.08.23.data.beta.pdet.csv'

# Download Pan Mammal GEO metadata
gse = geo.get_GEO(geo=GSE,
                  destdir="../data/SOFT")

metadata = gse.phenotype_data
metadata.to_csv('../data/GSE223748_panmammal_metadata.csv')

# Read metadata
df_meta = pd.read_csv('../data/GSE223748_panmammal_metadata.csv', index_col=0)

# Extract only metadata age
df_meta = df_meta[['title', 'characteristics_ch1.1.age']]
df_meta = df_meta.rename(columns={'characteristics_ch1.1.age': 'age'})
df_meta['tissue'] = 'Blood'
df_meta['organism'] = species

# import data as pandas dataframe
df_data = pd.read_csv(data_path, index_col=0)
df_data = df_data.filter(regex='beta')

# Replace column names with GSM
title_to_GSM_dict = dict(zip(df_meta['title'], df_meta.index))
new_columns = [title_to_GSM_dict[col_name.split('_')[0]] for col_name in list(df_data.columns)]
df_data = df_data.set_axis(new_columns, axis=1)

intersect_sample_ids = df_data.columns.intersection(df_meta.index)
intersect_data = df_data[intersect_sample_ids]
intersect_meta= df_meta.loc[intersect_sample_ids]

# Create AnnData Object
adata = ad.AnnData(intersect_data)
adata.var = intersect_meta
adata = adata[adata.obs.index.isin(mammalian_sites)]

# retrieve anage data from organism
anage_species = anage[(anage.Genus == genus) & (anage.Species == species)]
adata.uns['lifespan'] = anage_species['Maximum longevity (yrs)'].values[0]
adata.uns['common_name'] = anage_species['Common name'].values[0]

# filter samples without age information 
nan_samples = np.argwhere(adata.var.age.isnull()==False).flatten()
adata = adata[:, nan_samples].copy()

tissue_data = []

tissue_org_comb_list = adata.var.groupby(['tissue', 'organism']).size().index

for t_o_comb in tissue_org_comb_list:
    tissue, org = t_o_comb
    data = adata[:, (adata.var.tissue == tissue) & (adata.var.organism == org)].copy()
    data.uns['tissue'] = tissue

    # Fit  linear model through every site
    age_stack = np.broadcast_to(data.var.age, (data.shape)).T

    with pm.Model() as lr_model:
        s = pm.Flat('slope', shape=data.shape[0])
        i = pm.Flat('intercept',shape=data.shape[0])

        obs = pm.Normal('obs', mu=s*age_stack+ i, sigma = 0.01, observed=data.X.T)
        map_1 = pm.find_MAP(maxeval=500)

    # vectorised computation of pearson r
    r_values = r_regression(X=data.X.T, y=data.var.age)

    # Append linear information to AnnData
    data.obs['slope'] = map_1['slope']
    data.obs['intercept'] = map_1['intercept']
    data.obs['r'] = r_values

    data = evaluate_peaks (data)
    tissue_data.append(data)

    # Check whether the specified path exists or not
    path = (path_prefix 
            +'exports/')
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
    data.write_h5ad(path + '/' + org + '_' + tissue + '.h5ad')