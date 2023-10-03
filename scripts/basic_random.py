# %%
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
import os
from src.auxiliary_fcts_opt import *

from scipy.stats import linregress, norm
from sklearn.feature_selection import r_regression
from multiprocessing import Pool

# Parameters
sex_maturity_trim = False
init_min_samples = 15
drop_outliers = False
n_cores = 7
np.random.seed(123)
# set seed for reproducibilit
np.random.seed(123)

tissue = 'Blood'

with open(f'../exports/pan_mammal_{tissue}.pk', 'rb') as f:
    data_list = pickle.load(f)

for data in data_list:
    # append sexual maturity data where missing
    if data.uns['organism']=='Cervus canadensis':
        data.uns['common_name'] = 'Wapity elk'
        data.uns['lifespan'] = 25
        data.uns['sex_maturity'] = 2

    if data.uns['organism']=='Cervus elaphus':
        data.uns['common_name'] = 'Red deer'
        data.uns['lifespan'] = 31.5
        data.uns['sex_maturity'] = 2.17

results_path = f'../results/{tissue}/random_simulation/'
# Check if the directory exists
if not os.path.exists(results_path):
    # If it doesn't exist, create it
    os.makedirs(results_path)

# Replace methylation data with random non-linear methylation data
def methyl_data (data):
    """Create synthetic data randomly sapling from 
    the collection of orginal slopes from all datasets used in the study"""

    t = np.broadcast_to(data.var.age, (data.shape)).T
    slope = np.random.normal(scale=0.15, size=data.shape[0])
    intercept = np.random.uniform(low=0, high=1, size=data.shape[0])

    # slope = np.random.normal(0, 0.01, size=data.shape[0])
    mean = slope*t + intercept
    sigma = np.abs(np.random.normal(0.02,0.001, size=data.shape[0]))

    meth_data = norm.rvs(loc=mean, scale=sigma)
    meth_data = np.where(meth_data<0, 0, meth_data)
    meth_data = np.where(meth_data>1, 1, meth_data)

    return meth_data

# replace data with simulated data
for i, data in enumerate(data_list):
    data_list[i].var.age = np.random.uniform(low=0,
                                             high=data.uns['lifespan'],
                                             size=data.shape[1])
    data_list[i].X = methyl_data(data).T

# Trim to sexual maturity age range
def mod_data_trim (data_list):
    return data_trim(data_list,
                     sex_maturity_trim=sex_maturity_trim,
                     drop_outliers=drop_outliers)

with Pool(n_cores) as p:
    data_list = list(
        tqdm(p.imap(mod_data_trim, data_list), total=len(data_list)))

with Pool(n_cores) as p:
    data_list = list(
        tqdm(p.imap(compute_slopes, data_list), total=len(data_list)))

# compute r for each site
for data in data_list:    
    data.obs['r'] = r_regression(X=data.X.T, y=data.var.age)

scaling_law_list = []
r2_threshold_list = np.linspace(0,0.3, 101)
mean_sites_list = []
min_sites_list = []
for r2_threshold in tqdm(r2_threshold_list):
    slopes = [np.abs(data[data.obs.r**2>r2_threshold].obs.slope) for data in data_list]
    median_slopes = [np.median(s) for s in slopes]
    lifespan = [data.uns['lifespan'] for data in data_list]
    reg = linregress(x=np.log10(lifespan),
                        y=np.log10(median_slopes))

    scaling_law_list.append(reg.slope)
    mean_sites_list.append(np.mean([len(s) for s in slopes]))
    min_sites_list.append(np.min([len(s) for s in slopes]))


fig, ax = plt.subplots()
ax2 = plt.twinx()
sns.lineplot(x=r2_threshold_list, y=scaling_law_list, color=colors[0], legend=False, ax = ax, label='scaling law')
sns.lineplot(x=r2_threshold_list, y =mean_sites_list, ax=ax2, color=colors[1], label='mean cpgs')
sns.lineplot(x=r2_threshold_list, y =min_sites_list, ax=ax2, color=colors[2], label='min cpgs')
sns.despine(top=True, right=False, left=False, bottom=False)
# ax2.set_yscale('log')

# ask matplotlib for the plotted objects and their labels
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)
ax.set_ylabel('Scaling law')
ax.set_xlabel('r2 threshold')
ax2.set_ylabel('CpG count')
ax2.set_yscale('log')
plt.savefig(results_path + f'scaling_law_density_basic_random_{tissue}.png')
plt.savefig(results_path + f'scaling_law_density_basic_random_{tissue}.svg')
plt.show()
plt.clf()

r2_threshold = 0.1

median_slopes = [
        np.abs(data[data.obs.r**2>r2_threshold].obs.slope).median()
        for data in data_list]

lifespan = [data.uns['lifespan'] for data in data_list]
reg = linregress(x=np.log10(lifespan),
                    y=np.log10(median_slopes))
sns.regplot(x=np.log(lifespan), y=np.log(median_slopes))

slope = reg.slope
r2 = reg.rvalue**2

# Plot figure
fig, ax = plt.subplots()
sns.regplot(x=np.log10(lifespan),
            y=np.log10(median_slopes),
            ax=ax,
            scatter=False
            )

sns.scatterplot(x=np.log10(lifespan),
                    y=np.log10(median_slopes),
                    color='Black',
                    ax=ax)
se = reg.stderr

# Multiply by 1.96 to get 95% confidence interval
ci = 1.96*se

# Get the upper and lower bounds
upper = slope + ci
lower = slope - ci

plt.title(f'Slope: {slope: .2f} 95%CI {lower: .2f} to {upper: .2f} ---- r2: {r2: .2f}')
sns.despine()
plt.savefig(results_path + f'scaling_law_basic_random_{tissue}.png')
plt.savefig(results_path + f'scaling_law_basic_random_{tissue}.svg')
plt.show()
plt.clf()


# %%
