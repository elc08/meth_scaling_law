# Inference of scaling between methylation rates and lifespan in mammals

## General overview
This repository aims to develop tools to infer scaling laws of methylation rates and lifespan in conserved age-related CpG sites across mammals.
In particular we develop a statistically robust framework that allows the comparison of rates in a bounded system, such as methylation, in mammals of different lifespans.
The results of this analysis can be found in https://www.biorxiv.org/content/10.1101/2023.05.15.540689v1.abstract: 

## Data preparation
Download the following datasets and extract files in "data/"

- AnAge database:
    https://genomics.senescence.info/species/dataset.zip

- Mammalian Methylation Consortium dataset GSE223748:
    https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE223748&format=file&file=GSE223748%5FdatBetaNormalized%2Ecsv%2Egz

- Pan Troglodite dataset GSE136296:
    https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE136296&format=file

## Analysis
To run the full analysis and export all plots included in the paper run the following steps

1 . Create conda environment
```
conda env create -f env/meth_scaling_law.yml
```
or alternatively, for a more efficient management of environments, use mamba
```
mamba env create -f env/meth_scaling_law.yml
```
2. Activate conda environment
```
conda activate scaling_law_env
```

3. Run the full pipeline from base folder using
```
python full_run.py
```