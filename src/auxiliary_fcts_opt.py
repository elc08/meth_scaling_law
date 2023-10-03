
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

import pymc as pm
import numpy as np
from sklearn.feature_selection import r_regression
import seaborn as sns

from scipy.stats import linregress
import matplotlib.pyplot as plt

# List of domesticated animals
domesticated_list = ['Pig',
    'Sheep',
    'Cattle',
    'Horse',
    'Dog',
    'Cat','Winsconsin miniature pig']

# List of bat data
bat_list = [
 'Greater mouse-eared bat'
 ]

bat_list = bat_list + domesticated_list


def data_trim (data, sex_maturity_trim=True, drop_outliers=False):
    # append sexual maturity data where missing
    if data.uns['organism']=='Cervus canadensis':
        data.uns['common_name'] = 'Wapity elk'
        data.uns['lifespan'] = 25
        data.uns['sex_maturity'] = 2

    if data.uns['organism']=='Cervus elaphus':
        data.uns['common_name'] = 'Red deer'
        data.uns['lifespan'] = 31.5
        data.uns['sex_maturity'] = 2.17
        
    if sex_maturity_trim is True:
        # # keep samples above sexual maturity 
        # # and recompute slopes if necessary
        # data.uns['sex_maturity'] = data.uns['lifespan']*0.1
        data_trim = data[:, data.var.age>=data.uns['sex_maturity']].copy()
        data_trim.var.age = data_trim.var.age - data_trim.uns['sex_maturity']
    
    else:
        data_trim = data.copy()

    if drop_outliers is True:
        data_trim = data_trim[:, data_trim.var.outlier == False].copy()

    # save new data
    return data_trim

def set_age_trim(data_list, opt_prop=0.01):

    # find maximum ref and next trimming ages 
    ref_idx = [0]
    i = 1
    while i < len(data_list):
        ref_data = data_list[ref_idx[-1]]
        next_data = data_list[i]

        max_ref_age, max_next_age = find_optimal_gap(ref_data, next_data, opt_prop)
        if (ref_data[:, ref_data.var.age<=max_ref_age].shape[1]<15 
            or next_data[:, next_data.var.age<=max_next_age].shape[1]<15):
            i = i+1
            continue

        ref_data.uns['max_ref_age'] = max_ref_age
        next_data.uns['max_next_age'] = max_next_age
        ref_mammal = ref_data.uns['common_name']
        next_mammal = next_data.uns['common_name']
        gap = max_ref_age - max_next_age
        print(f'ref: {ref_mammal} -- next: {next_mammal} -- GAP:  {np.round(gap)} --')
        print(f'max_ref_age: {max_ref_age} -- total samples {ref_data[:, ref_data.var.age<max_ref_age].shape[1]}')
        print(f'max_next_age: {max_next_age} -- total samples {next_data[:, next_data.var.age<max_next_age].shape[1]}')

        ref_idx.append(i)
        i = i+1
    data_list = [data_list[i] for i in ref_idx]

    # Append information for first and last species
    data_list[0].uns['max_next_age'] = data_list[0].var.age.max()
    data_list[-1].uns['max_ref_age'] = data_list[-1].var.age.max()

    return data_list

def find_optimal_gap(ref_data, next_data, opt_prop):

    min_lifespan = min(ref_data.uns['lifespan'],
                                next_data.uns['lifespan'])

    ref_age_list = ref_data.var.sort_values(by='age', ascending=False).age.unique()
    for max_ref_age in ref_age_list:
        arg_optm_age_trim = np.argmin(np.abs(next_data.var.age - max_ref_age))
        next_age = next_data.var.iloc[arg_optm_age_trim].age
        max_next_age = next_data[:, next_data.var.age<=next_age].var.age.max()

        age_gap = np.abs(max_next_age-max_ref_age)
        min_diff = min(opt_prop*min_lifespan, 2)
        if age_gap < min_diff:
            break

    return max_ref_age, max_next_age


def vectorised_regression (data, progressbar=False):

    # Fit  linear model through every site in data
    age_stack = np.broadcast_to(data.var.age, (data.shape)).T

    with pm.Model() as lr_model:
        s = pm.Flat('slope', shape=data.shape[0])
        i = pm.Flat('intercept',shape=data.shape[0])

        # define prior for StudentT degrees of freedom
        # InverseGamma has nice properties:
        # it's continuous and has support x ∈ (0, inf)
        nu = pm.InverseGamma("nu", alpha=1, beta=1)

        # # define Student T likelihood
        # likelihood = pm.StudentT(
        #     "likelihood", mu=s*age_stack+i, sigma=0.01, nu=1, observed=data.X.T
        # )
        obs = pm.Normal('obs', mu=s*age_stack+ i, sigma = 0.01, observed=data.X.T)

        map = pm.find_MAP( maxeval=500, progressbar=progressbar)

    # vectorised computation of pearson r
    r_values = r_regression(X=data.X.T, y=data.var.age)

    return map['slope'], map['intercept'], r_values

def compute_slopes(data):
    """Compute slopes and intercept for every CpG in data"""
    
    slope, intercept, r_values = vectorised_regression(data)
    data.obs['slope'] = slope
    data.obs['intercept'] = intercept

    return data

def pairwise_slope_ratio (ref_data, next_data,
                          r2_threshold=0.1,
                          slope_threshold=0.001,
                          ref_mean_trim=True,
                          next_mean_trim=True):

    if ref_mean_trim is True:
        # Filter sites with restricted extremal mean meth 
        # notice we don't need to filter the ref dataset 
        ref_data = ref_data[ref_data.obs.ref_mean_meth<0.9]
        ref_data = ref_data[ref_data.obs.ref_mean_meth>0.1]

    if next_mean_trim is True:
        next_data = next_data[next_data.obs.next_mean_meth<0.9]
        next_data = next_data[next_data.obs.next_mean_meth>0.1]

    # Filter sites with low slopes
    ref_data = ref_data[np.abs(ref_data.obs.ref_slope)>slope_threshold]
    next_data = next_data[
        np.abs(next_data.obs.next_slope)>slope_threshold]
    
    # Filter sites based on r2_threshold
    ref_data = ref_data[ref_data.obs.ref_r**2 >r2_threshold]
    next_data = next_data[next_data.obs.next_r**2>r2_threshold]

    # Extract indices
    idx_ref_data = ref_data.obs.index
    idx_next_data = next_data.obs.index

    # intersect sites with reference
    idx_intersection = idx_ref_data.intersection(idx_next_data)

    # extract slopes from reference and data animal 
    # in intersecting sites
    ref_data_slopes = ref_data[idx_intersection].obs.ref_slope
    next_data_slopes = next_data[idx_intersection].obs.next_slope

    # Convert slopes to booleans showing sign
    slope_sign_reference = ref_data_slopes>0
    slope_sign_data = next_data_slopes>0

    # find index of sites with matching slope signs
    matching_idx = np.argwhere(
        slope_sign_data == slope_sign_reference).flatten()

    # filter out slopes with mathing signs
    ref_data_slopes = ref_data_slopes[matching_idx]
    next_data_slopes = next_data_slopes[matching_idx]

    slopes = next_data_slopes/ref_data_slopes
        
    return slopes

def compute_scaling_law(data_list, r2_threshold=0.1,
                        weight=False,
                        slope_threshold=0,
                        ref_mean_trim=True,
                        next_mean_trim=True,
                        plot_law=False,
                        count_sites=False,  
                        return_data=False):

    pairwise_median_slopes = [1]
    data_list[0].uns['pairwise_slope'] = pairwise_median_slopes[0]
    slopes_list = []
    for i in range(len(data_list)-1):
        # keep track of species used in analysis
        ref_data = data_list[i]
        next_data = data_list[i+1]

        slopes = pairwise_slope_ratio(ref_data, next_data,
                                      slope_threshold=slope_threshold,
                                      r2_threshold=r2_threshold,
                                      ref_mean_trim=ref_mean_trim,
                                      next_mean_trim=next_mean_trim)
        slopes_list.append(slopes)
        pairwise_median_slopes.append(np.median(slopes))
        if count_sites is True:
            if i == 0:
                min_sites_count = len(slopes)
                mean_sites_count = len(slopes)


            else:
                min_sites_count = np.min([min_sites_count,
                                            len(slopes)])
                mean_sites_count = np.mean([mean_sites_count,
                                            len(slopes)])
                
    # Extract pairwise estimators
    pairwise_estimator = np.cumprod(pairwise_median_slopes)
    law_data = [data.uns['lifespan'] for data in data_list]
    if weight is True:
        law_data = [data.var.average_adult_weight.iloc[0]
                    for data in data_list]

        #filter nan values
        not_nan = np.argwhere(~np.isnan(law_data)).flatten()
        law_data = np.array(law_data)
        law_data = law_data[not_nan]
        pairwise_estimator = pairwise_estimator[not_nan]


    # Linear regression
    reg = linregress(x=np.log10(law_data),
                    y=np.log10(pairwise_estimator))
    if return_data is True:
        return law_data, pairwise_estimator
    
    if plot_law is True:
        sns.set_theme(style='ticks')
        sns.set_palette(colors)

        # Linear regression
        reg = linregress(x=np.log10(law_data),
                        y=np.log10(pairwise_estimator))
        slope = reg.slope
        r2 = reg.rvalue**2
        print(reg.intercept)

        # Plot figure
        fig, ax = plt.subplots()
        sns.regplot(x=np.log10(law_data),
                            y=np.log10(pairwise_estimator),
                            ax=ax,
                            scatter=False
                            )

        sns.scatterplot(x=np.log10(law_data),
                            y=np.log10(pairwise_estimator),
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

        return fig, ax
    
    if count_sites is True:
        return reg.slope, min_sites_count, mean_sites_count
    else:
        return reg.slope


def ref_next_slopes (data):
    
    ref_data = data[:, data.var.age <= data.uns['max_ref_age']].copy()
    next_data = data[:, data.var.age <= data.uns['max_next_age']].copy()
    
    # Recompute slopes
    slope, intercept, rvalue = vectorised_regression(ref_data)
    data.obs['ref_slope'] = slope
    data.obs['ref_intercept'] = intercept
    data.obs['ref_r'] = rvalue
    data.obs['ref_mean_meth'] = ref_data.X.mean(axis=1)

    # Recompute slopes
    slope, intercept, rvalue = vectorised_regression(next_data)
    data.obs['next_slope'] = slope
    data.obs['next_intercept'] = intercept
    data.obs['next_r'] = rvalue
    data.obs['next_mean_meth'] = next_data.X.mean(axis=1)

    return data

def find_pairwise_idx(data_list, min_samples, opt_filter=True):
    ref_data = data_list[0]
    i = 1 
    drop_idx = []
    
    while i < len(data_list):
        # find next mammal
        next_data = data_list[i]
        # filter nex mammal data to max age of reference data
        max_ref_age = ref_data.var.age.max()

        ref_data.uns['max_ref_age'] = max_ref_age

        if opt_filter is True:
            # find nearest samples in next data in terms of max age

            arg_optm_age_trim = np.argmin(np.abs(next_data.var.age - max_ref_age))
            next_age = next_data.var.iloc[arg_optm_age_trim].age
            next_data = next_data[:, next_data.var.age<=next_age].copy()

        else: 
            next_data = next_data[:, next_data.var.age 
                                    <= max_ref_age*(1+opt_filter)].copy()
            
        # Check ages overlap between ref and next mammals
        next_sample_size = next_data.shape[1]

        while next_sample_size<min_samples:
            # print(ref_data.uns['common_name'],  nex_data.uns['common_name'])
            drop_idx.append(i)
            i = i+1
            
            if i >= len(data_list):
                break
            
            next_data = data_list[i]

            if opt_filter is True:
                # find nearest samples in next data in terms of max age

                arg_optm_age_trim = np.argmin(np.abs(next_data.var.age - max_ref_age))
                next_age = next_data.var.iloc[arg_optm_age_trim].age
                next_data = next_data[:, next_data.var.age<=next_age].copy()

            else: 
                next_data = next_data[:, next_data.var.age 
                                        <= max_ref_age*(1+opt_filter)].copy()
                
            # Check ages overlap between ref and next mammals
            next_sample_size = next_data.shape[1]

        # If maximum lifespan reached exit loop
        if i >= len(data_list):
            break

        ref_data = data_list[i]
        i +=1


    return drop_idx

def comute_pairwise_slopes(data_list, i, opt_filter=True):
    if i == 0:
        ref_data = data_list[0]
        return ref_data
    
    ref_data = data_list[i-1]
    next_data = data_list[i]

    # filter next mammal data to max age of reference data
    max_ref_age = ref_data.var.age.max()

    # find nearest samples in next data in terms of max age
    if opt_filter is True:
        arg_optm_age_trim = np.argmin(np.abs(next_data.var.age - max_ref_age))
        next_age = next_data.var.iloc[arg_optm_age_trim].age
        next_data_restricted = next_data[:, next_data.var.age<=next_age].copy()

    else: 
        next_data_restricted = next_data[:, next_data.var.age 
                                <= max_ref_age*(1+opt_filter)].copy()
        
    # set next_slope as slope from trimmed data
    slope, intercept, rvalue = vectorised_regression(next_data_restricted)
    next_data.obs['next_slope'] = slope
    next_data.obs['next_r'] = rvalue
    next_data.obs['next_intercept'] = intercept
    next_data.obs['next_mean_meth'] = next_data_restricted.X.mean(axis=1)

    return next_data
