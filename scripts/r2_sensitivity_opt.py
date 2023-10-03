# %%
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
import os
from src.auxiliary_fcts_opt import *

from scipy.stats import linregress, norm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from multiprocess import Pool

import plotly.graph_objects as go

if 'tissue' not in globals():
    tissue = 'Blood'  # Assign a value if tissue has not been defined

if 'n_cores' not in globals():
    n_cores = 7  # Assign a value if tissue has not been defined

print(f'Processing {tissue} data')

with open(f'../exports/pan_mammal_{tissue}_outliers.pk', 'rb') as f:
    data_list = pickle.load(f)

# data_list = [data for data in data_list if data.var.age.max()>0.3*data.uns['lifespan']]

# Parameters
sex_maturity_trim = True
ref_mean_trim = True
next_mean_trim = True
filter_bad_organisms = False
init_min_samples = 20
min_samples = 20
increasing = True
decreasing = True
drop_outliers = True

opt_prop = 0.02


min_r2_threshold=0
max_r2_threshold=0.2

print( f'Tissue: {tissue}\n' 
      + 'Model parameters: \n'
      + f'Sexual Maturity Trim: {str(sex_maturity_trim)}\n'
      + f'filter bad organisms: {str(filter_bad_organisms)}\n'
      + f'Opt_prop: {opt_prop}')

# create output directories
if sex_maturity_trim is False:
    results_path = f'../results/{tissue}/no_trim/'
    # Check if the directory exists
    if not os.path.exists(results_path):
        # If it doesn't exist, create it
        os.makedirs(results_path)
elif filter_bad_organisms is True:
    results_path = f'../results/{tissue}/no_bat/'
    # Check if the directory exists
    if not os.path.exists(results_path):
        # If it doesn't exist, create it
        os.makedirs(results_path)
else:
    results_path = f'../results/{tissue}/'
    # Check if the directory exists
    if not os.path.exists(results_path):
        # If it doesn't exist, create it
        os.makedirs(results_path)

# in blood add pan troglodytes
if tissue == 'Blood':
    chimp = ad.read_h5ad('../exports/troglodytes_Blood.h5ad')
    chimp.uns['organism'] = 'Pan troglodytes'
    chimp.uns['sex_maturity'] = 9.24
    chimp.var['average_adult_weight'] = 44.984
    chimp.var['outlier'] = False
    data_list.append(chimp)

    data_list = [data for data in data_list
                if data.uns['common_name'] not in 
                ['Winsconsin miniature pig']]    

# if tissue == 'Skin':
#     data_list = [data for data in data_list
#                  if data.uns['common_name'] not in 
#                  [ 'Greater mouse-eared bat']]

if filter_bad_organisms is True:
    if tissue == 'Blood':
        data_list = [data for data in data_list
                    if data.uns['common_name'] not in domesticated_list]
    if tissue == 'Skin':
        data_list = [data for data in data_list
            if data.uns['common_name'] not in bat_list]

# Add in some data that AnAge doesn't have
# Trim to sexual maturity age range
def mod_data_trim (data_list):
    return data_trim(data_list,
                     sex_maturity_trim=sex_maturity_trim,
                     drop_outliers=drop_outliers)

with Pool(n_cores) as p:
    data_list = list(
        tqdm(p.imap(mod_data_trim, data_list), total=len(data_list)))

# Sort data list by max observed age
data_list.sort(key=lambda x: x.var.age.max())

# # Filter out species with less than min samples
data_list = [data for data in data_list if data.shape[1]>=init_min_samples]

data_list = set_age_trim(data_list, opt_prop=opt_prop)

with Pool(n_cores) as p:
    data_list = list(
        tqdm(
            p.imap(ref_next_slopes,
                   data_list),
            total=len(data_list))
        )

# Export pickled object with computed pairwise slopes
with open(results_path + f'pan_mammal_{tissue}_pairwise.pk', 'wb') as f:
    pickle.dump(data_list, f)

def  r2_scaling_law (r2_threshold):
    return compute_scaling_law(data_list=data_list,
                    r2_threshold=r2_threshold,
                    slope_threshold=0,
                    ref_mean_trim=ref_mean_trim,
                    next_mean_trim=next_mean_trim,
                    count_sites=True,
                    plot_law=False)


r2_threshold_list = np.linspace(min_r2_threshold,max_r2_threshold, 101)

with Pool(n_cores) as pool:
    result = list(tqdm(pool.imap(r2_scaling_law, r2_threshold_list), total=len(r2_threshold_list)))

scaling_law_list, min_sites_list, mean_sites_list = zip(*result)

max_idx = np.argmin(np.abs(np.array(min_sites_list)-10))

scaling_law_list = scaling_law_list[:max_idx]
mean_sites_list = mean_sites_list[:max_idx]
min_sites_list = min_sites_list[:max_idx]
r2_threshold_list = r2_threshold_list[:max_idx]

# Scaling law plot
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
plt.savefig(results_path + f'scaling_law_stability_{tissue}.png')
plt.savefig(results_path + f'scaling_law_stability_{tissue}.svg')
plt.show()
plt.clf()

fig, ax = plt.subplots()
sns.kdeplot(scaling_law_list, ax=ax)

data = ax.lines[0].get_xydata()# %%
kde_map = data[np.where(data[:, 1] == max(data[:, 1]))]
scaling_law_map = kde_map[0][0]
r2_map_arg = np.nanargmin(np.abs(np.array(scaling_law_list)- scaling_law_map))
sns.scatterplot(x=[kde_map[0][0]], y=[kde_map[0][1]], color='red')
plt.axvline(scaling_law_map, 0, kde_map[0][1] , color='red')
ax.set_xlabel('r2 threshold')
sns.despine()
plt.savefig(results_path + f'scaling_law_density_{tissue}.png')
plt.savefig(results_path + f'scaling_law_density_{tissue}.svg')

plt.show()
plt.clf()

optimal_r2 = r2_threshold_list[r2_map_arg]

fig, ax = compute_scaling_law(data_list=data_list,
                    r2_threshold=optimal_r2,
                    slope_threshold=0,
                    ref_mean_trim=ref_mean_trim,
                    next_mean_trim=next_mean_trim,
                    count_sites=False,
                    return_data=False,
                    plot_law=True)

ax.set_xlabel('log - lifespan (yr)')
ax.set_ylabel('log - slope ratio')
plt.savefig(results_path + f'scaling_law_{tissue}.png')
plt.savefig(results_path + f'scaling_law_{tissue}.svg')

# Weight scaling law
fig, ax = compute_scaling_law(data_list=data_list,
                    r2_threshold=optimal_r2,
                    weight=True,
                    slope_threshold=0,
                    ref_mean_trim=ref_mean_trim,
                    next_mean_trim=next_mean_trim,
                    count_sites=False,
                    return_data=False,
                    plot_law=True)

ax.set_xlabel('log - adult weight (kg)')
ax.set_ylabel('slope ratio')
plt.savefig(results_path + f'weight_scaling_law_{tissue}.png')
plt.savefig(results_path + f'weight_scaling_law_{tissue}.svg')


# Weight scaling law
positive_data = []
negative_data = []

# Positive and negative slopes scaling laws
for data in data_list:
    positive_data.append(data[data.obs.ref_slope>=0].copy())
    negative_data.append(data[data.obs.ref_slope<0].copy())

fig, ax = compute_scaling_law(data_list=positive_data,
                r2_threshold=optimal_r2,
                slope_threshold=0,
                ref_mean_trim=ref_mean_trim,
                next_mean_trim=next_mean_trim,
                count_sites=False,
                return_data=False,
                plot_law=True)

ax.set_xlabel('log - lifespan (yr)')
ax.set_ylabel('log - slope ratio')
plt.savefig(results_path + f'scaling_law_{tissue}_positive.png')
plt.savefig(results_path + f'scaling_law_{tissue}_positive.svg')


plt.show()
plt.clf()


fig, ax = compute_scaling_law(data_list=negative_data,
                r2_threshold=optimal_r2,
                slope_threshold=0,
                ref_mean_trim=ref_mean_trim,
                next_mean_trim=next_mean_trim,
                count_sites=False,
                return_data=False,
                plot_law=True)

ax.set_xlabel('log - lifespan (yr)')
ax.set_ylabel('slope ratio')
plt.savefig(results_path + f'scaling_law_{tissue}_negative.png')
plt.savefig(results_path + f'scaling_law_{tissue}_negative.svg')

plt.show()
plt.clf()

lifespan, law = compute_scaling_law(data_list=data_list,
                r2_threshold=optimal_r2,
                slope_threshold=0,
                ref_mean_trim=ref_mean_trim,
                next_mean_trim=next_mean_trim,
                count_sites=False,
                return_data=True,
                plot_law=False)

fig = px.scatter(x=np.log(lifespan),
           y=np.log(law),
           trendline='ols',
           hover_name=[data.uns['common_name'] for data in data_list])
fig.write_html(results_path + 'scaling_law.html')

law_df = pd.DataFrame(columns=['lifespan', 'pairwise_slope'], data= np.array([lifespan, law]).T)
law_df['common_name'] = [data.uns['common_name'] for data in data_list]
law_df['average_adult_weight'] = [data.var['average_adult_weight'].unique()[0] for data in data_list]

law_df.to_csv(results_path + 'dataframe_scaling.csv')


# Compute comparison dataframe
comparison_df = pd.DataFrame(columns=['reference_mammal', 'comparison_mammal', 'max_ref_age', 'max_comp_age', 'ref_mammal_count', 'comp_mammal_count', 'median_slope_ratio', 'cumulative_slope'])
for i, data in enumerate(data_list):
    if i == 0:
        continue
    ref_data = data_list[i-1]
    ref_data_count = ref_data[:, ref_data.var.age < ref_data.uns['max_ref_age']].shape[1]
    next_data = data
    next_data_count = next_data[:, next_data.var.age < next_data.uns['max_next_age']].shape[1]

    median_slope = law_df.iloc[i]['pairwise_slope']/law_df.iloc[i-1]['pairwise_slope']
    comparison_df = pd.concat([comparison_df,
                               pd.DataFrame({'reference_mammal':[ref_data.uns['common_name']],
                                             'comparison_mammal':[next_data.uns['common_name']],
                                             'max_ref_age':[ref_data.uns['max_ref_age']], 
                                             'max_comp_age':[data.uns['max_next_age']],
                                             'ref_mammal_count':[ref_data_count],
                                             'comp_mammal_count':[next_data_count],
                                             'median_slope_ratio':[median_slope],
                                             'cumulative_slope': law_df.iloc[i]['pairwise_slope']})],
                               ignore_index=True)
    

comparison_df.to_csv(results_path + 'comparison_df.csv')
