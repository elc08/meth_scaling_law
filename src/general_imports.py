import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import cpu_count
import anndata as ad
import pandas as pd
import json
import pickle

# Pymc progress bar
from IPython.display import clear_output, DisplayHandle
def update_patch(self, obj):
    clear_output(wait=True)
    self.display(obj)
DisplayHandle.update = update_patch

class Organism:
  def __init__(self, name, common_name, tissue, longevity, adult_weight, metabolic_rate, data, species_data):
     self.name = name
     self.common_name = common_name
     self.tissue = tissue
     self.longevity = longevity
     self.adult_weight = adult_weight
     self.metabolic_rate = metabolic_rate
     self.data = data
     self.species_data = species_data

# sns.set_theme(style='ticks')

plt.rcParams['svg.fonttype'] = 'none'


sns_colors = sns.color_palette().as_hex()
# https://nanx.me/ggsci/reference/pal_npg.html
colors = [
    '#4DBBD5FF',    # blue
    '#E64B35FF',    # red
    '#00A087FF',    # dark green
    '#F39B7FFF',    # orenginh
    '#3C5488FF',    # dark blue
    '#8491B4FF',
    '#91D1C2FF',
    '#DC0000FF',
    '#B09C85FF',    # light grey
    '#7E6148FF',
]
sns.set_palette(sns.color_palette(colors))

# CON_PALLETE = sns.color_palette("blend:#E64B35,#4DBBD5")
# # plt.rc("axes.spines", top=False, right=False)

# Make sure pymc is displaying progressbars
from IPython.display import clear_output, DisplayHandle
def update_patch(self, obj):
    clear_output(wait=True)
    self.display(obj)
DisplayHandle.update = update_patch
