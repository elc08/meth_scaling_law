import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
import GEOparse as geo
from sklearn.feature_selection import r_regression
import pymc as pm

# Download Pan Mammal GEO metadata
gse = geo.get_GEO(geo='GSE223748',
                  destdir="../data/SOFT")

metadata = gse.phenotype_data

metadata.to_csv('../data/GSE223748_panmammal_metadata.csv')

# Read metadata
metadata = pd.read_csv('../data/GSE223748_panmammal_metadata.csv')

# Extract only columns of interest
metadata = metadata[metadata.columns[:18]]
metadata['id'] = metadata["title"].apply(lambda x: pd.Series(x.split()[-1]))
metadata = metadata.set_index('id')
metadata = metadata[metadata.columns[8:]]

# Rename columns
new_column_names = ['tissue', 'organism', 'tax_id', 'age',
                    'age_confidence', 'female',  'common_name', 'max_lifespan',
                    'average_adult_weight','age_at_sexual_maturity']
metadata.columns = new_column_names

# Read Data
data = pd.read_csv('../data/GSE223748_datBetaNormalized.csv', index_col=0)

# Merge data and metadata in single AnnData
index = metadata.index
adata = ad.AnnData(data[index], var=metadata)

# export anndata
adata.write_h5ad('../data/pan_mammal.h5ad')

# List tissue with data from more than 5 organisms
tissue_list = ['Blood', 'Skin']

# For each tissue create a list of AnnData Objects
for tissue in tqdm(tissue_list):
    # filter data by tissue
    tissue_data = adata[:, adata.var.tissue == tissue].copy()

    # extract list of organisms with tissue data
    organism_list = list(tissue_data.var.organism.unique())

    # create tissue and organism specific Anndata objects 
    # for organisms with more than 10 samples
    org_data_list = []
    for org in organism_list:
        org_data = tissue_data[:, tissue_data.var.organism == org]
        # filter samples with nan age
        not_nan_age = np.argwhere(~np.isnan(org_data.var.age)).flatten()
        org_data = org_data[:, not_nan_age]

        # skip organism if too little samples are present
        if org_data.shape[1] < 10:
            continue

        # Append unstructured organism information
        org_data.uns['organism'] = org_data[:, 0].var['organism'].values[0]
        org_data.uns['common_name'] = org_data[:, 0].var['common_name'].values[0]
        org_data.uns['lifespan'] = org_data[:, 0].var['max_lifespan'].values[0]
        org_data.uns['sex_maturity'] = org_data[:, 0].var['age_at_sexual_maturity'].values[0]

        # append sexual maturity and lifespan data where missing
        if org_data.uns['organism']=='Cervus canadensis':
            org_data.uns['common_name'] = 'Wapity elk'
            org_data.uns['lifespan'] = 25
            org_data.uns['sex_maturity'] = 2

        if org_data.uns['organism']=='Cervus elaphus':
            org_data.uns['common_name'] = 'Red deer'
            org_data.uns['lifespan'] = 31.5
            org_data.uns['sex_maturity'] = 2.17
            
        org_data_list.append(org_data)

    with open(f'../exports/pan_mammal_{str(tissue)}.pk', 'wb') as f:
        pickle.dump(org_data_list, f)