from scipy.signal import find_peaks
import statsmodels.api as sm
import numpy as np

def evaluate_peaks(adata):
    peak_list = []
    for site in adata:
        site_data = site.X.flatten()
        dens = sm.nonparametric.KDEUnivariate(site_data)
        dens.fit(kernel='gau', bw=0.05)
        x = np.linspace(-0.1,1.1,100) #restrict range to (0,1)
        y = dens.evaluate(x)
        y = y/max(y)
        peak_list.append(
            find_peaks(y, prominence=0.2)[0].shape[0])
        # peak_list.append(
        #     find_peaks(y, prominence=1)[0].shape[0])
        
    adata.obs['n_peaks'] = peak_list
    return adata