# SynProtX

## Setting up environment

We use Conda to manage Python dependencies in this project. To reproduce our environment, please run the following script in the terminal:

```sh
conda env create --name SynProtX --file env.yml
conda activate SynProtX
```

## Training and testing

To execute a training and testing task for our model, run the following script

```sh
python synprotx/<model>.py -d <database> -m <mode>
```

Possible options are listed below

- `model` represents the name of the model to run. Must be one of `gat`, `gcn`, `attentivefp` and `gatfp`.
- `--database`/`-d` specifies data source to train the model on. Must be one of `almanac-breast`, `almanac-lung`, `almanac-ovary`, `almanac-skin`, `friedman`.
- `--mode`/`-m` input must be either `clas`, for classification task, or `regr`, for regression task. Default to `clas`
- Flags `--no-feamol`, `--no-feagene`, `--no-feaprot` disable the molecule, gene expression and protein expression features, respectively, when propagate through the model.

Execute `python  synprotx/<model>.py -h` for more detailed description.
