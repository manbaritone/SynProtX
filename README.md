# SynProtX

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

Datasets, hyperparameters, and model checkpoints can be downloaded through [![https://doi.org/10.5281/zenodo.13285494](https://zenodo.org/badge/DOI/10.5281/zenodo.13285494.svg)](https://doi.org/10.5281/zenodo.13285494).

## Using SynProtX for inference

SynProtX allows the prediction of synergistic effects between drug combinations through inference using the SynProtX model. It leverages various tissue-specific and study datasets to make these predictions.

### Usage Example

To perform inference, you can run the following command:

```bash
python synprotx_inference.py --smi1 "CCOc1ccc2c(c1)N=C(N)N(c3ccc(Cl)cc3)S2" --smi2 "CN1CCC(CC1)Nc2nccc3c2ncn3C" \
--dataset ALMANAC-Breast --cell_line MCF7 --task classification --thr 0.5
```

In this example:
- `--smi1` and `--smi2` represent the SMILES strings of the two drug compounds being tested.
- `--dataset` specifies the dataset to use (e.g., ALMANAC-Breast).
- `--cell_line` indicates the cell line to consider (e.g., MCF7).
- `--task` defines the type of task: classification for synergy/antagonism prediction or regression for raw score prediction.
- `--thr` sets the threshold for classification tasks, used to differentiate between synergistic and antagonistic interactions.

### Available Options

| Option        | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| `--smi1`      | SMILES string of the first compound (required)                              |
| `--smi2`      | SMILES string of the second compound (required)                             |
| `--cell_line` | Cell-line identifier (e.g. MCF7) (required)                                 |
| `--dataset`   | Dataset to use (default: `ALMANAC-Breast`). Available options are:          |
|               | - For Tissue Datasets: `ALMANAC-Breast`, `ALMANAC-Lung`, `ALMANAC-Ovary`, `ALMANAC-Skin` |
|               | - For Study Datasets: `FRIEDMAN`, `ONEIL`                                    |
| `--task`      | Task type (default: `regression`). Options:                                 |
|               | - `classification`, `regression`                                             |
| `--device`    | Device for computation (default: `cpu`). Options:                           |
|               | - `cpu`, `cuda:0` (or another CUDA device string)                            |
| `--thr`       | Threshold for classifying synergy vs antagonism (only for `classification` task). Default: `0.5` |

## Dataset availability

| Dataset          | Cell Lines                                                                 |
|------------------|-----------------------------------------------------------------------------|
| ALMANAC-Breast   | BT-549, MCF7, MDA-MB-231, MDA-MB-468                                        |
| ALMANAC-Lung     | A549, EKVX, HOP-62, HOP-92, NCI-H226, NCI-H460, NCI-H522                    |
| ALMANAC-Ovary    | OVCAR-4, OVCAR-5, OVCAR-8, SK-OV-3                                          |
| ALMANAC-Skin     | SK-MEL-2, SK-MEL-5, SK-MEL-28, UACC-257                                     |
| FRIEDMAN (Skin)  | A2058, G-361, IPC-298, RVH-421, SK-MEL-2, SK-MEL-5, SK-MEL-28, UACC-257    |
| ONEIL (Several Tissues) | A2058 (skin), NCI-H460 (lung), SK-OV-3 (ovary), A2780 (ovary), A427 (lung), RKO (large intestine), SW837 (large intestine) |

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

**Note:** There are more options to configure. Execute `python synprotx/<model>.py -h` for a more detailed description.

## Results

The performance evaluation per repeated fold can be looked up in the folder "results". This folder includes a comprehensive list of all results files obtained from the training process.
The models in comparison are  `XGBoost`, `DeepDDS`, `DeepSyn`, `SynProtX` variations, and `AttenSyn`. The type of split includes `random`, cold-start for (leave-one-out) drugs, drug combinations,
and cell lines, and ablation (gene and protein) on both classification and regression tasks.

> **Disclaimer:** The CSV files in the "results" folder are not covered by the same MIT license as the source code.
These data files are dedicated to the public domain under **CC0**.

## Citations

**Research Article**
```bibtex
@article{boonyarit2025synprotx_gigascience,
  author       = {Boonyarit, Bundit and
                  Kositchutima, Matin and
                  Na Phattalung, Tisorn and
                  Yamprasert, Nattawin and
                  Thuwajit, Chanitra and
                  Rungrotmongkol, Thanyada and
                  Nutanong, Sarana},
  title        = {SynProtX: A Large-Scale Proteomics-Based Deep Learning Model for Predicting Synergistic Anticancer Drug Combinations},
  journal      = {GigaScience},
  volume       = {},
  number       = {},
  pages        = {},
  keywords     = {cancer drug combination, deep learning, drug discovery, graph neural networks, machine learning, multi-omics, personalized medicine, proteomics, synergistic effect},
  doi          = {https://doi.org/10.1093/gigascience/giaf080},
  url          = {https://doi.org/10.1093/gigascience/giaf080}
}

@article{boonyarit2025synprotx_zenodo,
  author       = {Boonyarit, Bundit and
                  Kositchutima, Matin and
                  Na Phattalung, Tisorn and
                  Yamprasert, Nattawin and
                  Thuwajit, Chanitra and
                  Rungrotmongkol, Thanyada and
                  Nutanong, Sarana},
  title        = {SynProtX: A Large-Scale Proteomics-Based Deep Learning Model for Predicting Synergistic Anticancer Drug Combinations},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.15603481},
  url          = {https://doi.org/10.5281/zenodo.15603481},
  note         = {[Dataset]}
}
```

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
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.15603481},
  url          = {https://doi.org/10.5281/zenodo.15603481},
  note         = {[Dataset]}
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
  publisher    =  {WorkflowHub}
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
  url          = {https://registry.dome-ml.org/review/7hk5upi8vx}
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
  url          = {https://archive.softwareheritage.org/swh:1:snp:750d09d4ed20b1628cef1f20cf0d2b2e518c4a3b;origin=https://github.com/manbaritone/SynProtX}
}
```
