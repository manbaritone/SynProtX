# SynProtX

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13285494.svg)](https://doi.org/10.5281/zenodo.13285494)

An official implementation of our research paper **"SynProtX: A Large-Scale Proteomics-Based Deep Learning Model for Predicting Synergistic Anticancer Drug Combinations"**.

SynProtX is a deep learning model that integrates large-scale proteomics data, molecular graphs, and chemical fingerprints to predict synergistic effects of anticancer drug combinations. It provides robust performance across tissue-specific and study-specific datasets, enhancing reproducibility and biological relevance in drug synergy prediction.

![SynProtX architecture](https://github.com/manbaritone/SynProtX/blob/main/synprotx_architect.png)

## DOME-ML

**SynProtX** has been registered in the  DOME Registry to promote transparency, best practices, and reproducibility in supervised machine learning for biology. The registry follows the DOME framework, which emphasizes four key pillars:

- **Data:** Clear documentation of dataset composition and partitioning
- **Optimization:** Transparent description of training procedures and hyperparameters
- **Model:** Detailed architecture, weight files, and implementation metadata
- **Evaluation:** Comprehensive reporting of performance metrics and validation strategy

**To view the full registration:** https://registry.dome-ml.org/review/7hk5upi8vx.

If you are referencing SynProtX in a publication, please include the citation below.

## Setting up environment

We use [Miniconda](https://docs.anaconda.com/miniconda/) to manage Python dependencies in this project. To reproduce our environment, please run the following script in the terminal:

```sh
conda env create -f env.yml
conda activate SynProtX
```

## Downloading raw data

Datasets, hyperparameters, and model checkpoints can be downloaded through [Zenodo](https://doi.org/10.5281/zenodo.13285494).

## Generating dataset

A tarball will be obtained after download. After file extraction, move all nested folders to the root of this project directory. You might need to move all files in `data/export` up to `data` folder. Otherwise, you will run the Jupyter Notebook files to generate mandatory data. Let’s take a look at `ipynb` folder. Run the following files in order if you want to replicate our exported data.

- `01_drugcomb_clean.ipynb` → `cleandata_cancer.csv`
- `02_CCLE_gene_expression` → `CCLE_expression_cleaned.csv`
- `03_omics_preprocess` → `protein_omics_data_cleaned.csv`
- `04_drugcomb_gene_prot_clean` → `data_preprocessing_gene.pkl`, `data_drugcomb.pkl`, `data_preprocessing_protein.pkl`
- `05_graph_generate.ipynb` → `nps_intersected` folder
- `06_smiles_feat_generate.ipynb` → `smiles_graph_data.pkl`
- `07_to_ecfp6_deepsyn.ipynb` → `deepsyn_drug_row.npy`, `deepsyn_drug_col.npy`

> If the console shows an error indicating that SMILES are not found, you MUST run the file `06_smiles_feat_generate.ipynb` again to regenerate data.

## Training and testing

To execute a training and testing task for our model, run the following script

```sh
python synprotx/<model>.py -d <database> -m <mode>
```

Possible options are listed below.

- `model` represents the name of the model to run. Must be one of `gat`, `gcn`, `attentivefp` and `gatfp`.
- `--database`/`-d` specifies data source to train the model on. Must be one of `almanac-breast`, `almanac-lung`, `almanac-ovary`, `almanac-skin`, `friedman`, `oneil`.
- `--mode`/`-m` input must be either `clas`, for classification task, or `regr`, for regression task. Default to `clas`
- Flags `--no-feamol`, `--no-feagene`, `--no-feaprot` disable the molecule branch, gene expression branch, and protein expression branch, respectively, when propagate through the model.

**Note:** There are more options to configure. Execute `python  synprotx/<model>.py -h` for a more detailed description.

## Results

The performance evaluation per repeated fold can be looked up in the folder "results". This folder includes a comprehensive list of all results files obtained from the training process.
The models in comparison are  `XGBoost`, `DeepDDS`, `DeepSyn`, `SynProtX` variations, and `AttenSyn`. The type of split includes `random`, cold-start for (leave-one-out) drugs, drug combinations,
and cell lines, and ablation (gene and protein) on both classification and regression tasks.

> **Disclaimer:** The CSV files in the "results" folder are not covered by the same MIT license as the source code.
These data files are dedicated to the public domain under **CC0 public domain**.

## Citations

**Zenodo**
```bibtex
@online{boonyarit2025synprotx_zenodo,
  author       = {Boonyarit, Bundit and
                  Kositchutima, Matin and
                  Na Phattalung, Tisorn and
                  Yamprasert, Nattawin and
                  Thuwajit, Chanitra and
                  Rungrotmongkol, Thanyada and
                  Nutanong, Sarana},
  title        = {SynProtX: A Large-Scale Proteomics-Based Deep Learning Model for Predicting Synergistic Anticancer Drug Combinations},
  year         = {2025},
  note         = {[Data set]},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.15603481},
  url          = {https://doi.org/10.5281/zenodo.15603481},
  note         = {[Dataset]},
}
```

**WorkflowHub**
```bibtex
@online{boonyarit2025synprotx_workflowhub,
  author       =  {Boonyarit, Bundit and
                  Kositchutima, Matin and
                  Na Phattalung, Tisorn and
                  Yamprasert, Nattawin and
                  Thuwajit, Chanitra and
                  Rungrotmongkol, Thanyada and
                  Nutanong, Sarana},
  title        =  {SynProtX},
  year         =  {2025}
  url          =  {https://workflowhub.eu/workflows/1726?version=3},
  DOI          =  {10.48546/WORKFLOWHUB.WORKFLOW.1726.3},
  publisher    =  {WorkflowHub},
}
```

**DOME-ML**
```bibtex
@online{boonyarit2025synprotx_dome,
  author       = {Boonyarit, Bundit and
                  Kositchutima, Matin and
                  Na Phattalung, Tisorn and
                  Yamprasert, Nattawin and
                  Thuwajit, Chanitra and
                  Rungrotmongkol, Thanyada and
                  Nutanong, Sarana},
  title        = {SynProtX: A Large-Scale Proteomics-Based Deep Learning Model for Predicting Synergistic Anticancer Drug Combinations},
  year         = {2025},
  note         = {[DOME-ML Annotations]},
  url          = {https://registry.dome-ml.org/review/7hk5upi8vx},
}

```

**Software Heritage**
```bibtex
@online{boonyarit2025synprotx_software,
  author       = {Boonyarit, Bundit and
                  Kositchutima, Matin and
                  Na Phattalung, Tisorn and
                  Yamprasert, Nattawin and
                  Thuwajit, Chanitra and
                  Rungrotmongkol, Thanyada and
                  Nutanong, Sarana},
  title        = {SynProtX: A Large-Scale Proteomics-Based Deep Learning Model for Predicting Synergistic Anticancer Drug Combinations (Version 1)},
  year         = {2025},
  note         = {[Computer software]},
  url          = {https://archive.softwareheritage.org/swh:1:snp:28019112cc4ea0fdcb8c529e6c895b0dcc434add;origin=https://github.com/manbaritone/SynProtX},
}
```
