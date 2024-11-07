# SynProtX: Enhanced Deep Learning Model for Predicting Anticancer Drug Synergy Using Large-Scale Proteomics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14040426.svg)](https://doi.org/10.5281/zenodo.14040426)

An official implementation of our reasearch paper "SynProtX: Enhanced Deep Learning Model for Predicting Anticancer Drug Synergy Using Large-Scale Proteomics".

## Setting up environment

We use [Miniconda](https://docs.anaconda.com/miniconda/) to manage Python dependencies in this project. To reproduce our environment, please run the following script in the terminal:

```sh
conda env create --name SynProtX --file env.yml
conda activate SynProtX
```

## Downloading raw data

Dataset, hyperparameters, and model checkpoints can be download through [Zenodo](https://doi.org/10.5281/zenodo.14040426).

## Generating dataset

A tarball will be obtained after download. After file extraction, move all nested folders to the root of this project directory. You might need to move all files in `data/export` up to `data` folder. Otherwise, you will run the Jupyter Notebook files to generate mandatory data. Let’s take a look at `ipynb` folder. Run the following files in order if you want to replicate our exported data.

- `01_drugcomb_clean.ipynb` → `cleandata_cancer.csv`
- `02_CCLE_gene_expression` → `CCLE_expression_cleaned.csv`
- `03_omics_preprocess` → `protein_omics_data_cleaned.csv`
- `04_drugcomb_gene_prot_clean` → `data_preprocessing_gene.pkl`, `data_drugcomb.pkl`, `data_preprocessing_protein.pkl`
- `05_graph_generate.ipynb` → `nps_intersected` folder
- `06_smiles_feat_generate.ipynb` → `smiles_graph_data.pkl`
- `07_to_ecfp6_deepsyn.ipynb` → `deepsyn_drug_row.npy`, `deepsyn_drug_col.npy`

> If the console shows an error indicating that SMILES not found, you MUST run the file `06_smiles_feat_generate.ipynb` again to regenerate data.

## Training and testing

To execute a training and testing task for our model, run the following script

```sh
python synprotx/<model>.py -d <database> -m <mode>
```

Possible options are listed below.

- `model` represents the name of the model to run. Must be one of `gat`, `gcn`, `attentivefp` and `gatfp`.
- `--database`/`-d` specifies data source to train the model on. Must be one of `almanac-breast`, `almanac-lung`, `almanac-ovary`, `almanac-skin`, `friedman`.
- `--mode`/`-m` input must be either `clas`, for classification task, or `regr`, for regression task. Default to `clas`
- Flags `--no-feamol`, `--no-feagene`, `--no-feaprot` disable the molecule branch, gene expression branch, and protein expression branch, respectively, when propagate through the model.

Note that there are more options to configure. Execute `python  synprotx/<model>.py -h` for more detailed description.
