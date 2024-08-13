from __future__ import division
from __future__ import print_function

import os
import gc
from dgl import batch, load_graphs
from typing import Optional
from torchmetrics.functional.classification import (
    binary_auroc,
    binary_precision_recall_curve,
    binary_f1_score,
    binary_matthews_corrcoef,
    binary_cohen_kappa,
)
from torchmetrics.functional.classification import (
    binary_precision,
    binary_recall,
    binary_accuracy,
    binary_specificity,
)
from torchmetrics.functional import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    pearson_corrcoef,
    spearman_corrcoef,
)
from dgllife.model.gnn import AttentiveFPGNN
from dgllife.model.readout import AttentiveFPReadout
import pandas as pd
import numpy as np
import csv
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from deepchem.splits import RandomSplitter
from torch_geometric.data import Data
from torch_geometric.loader import DataListLoader
from torch_geometric.nn import DataParallel
from torch_geometric.nn.models import AttentiveFP
import random
import deepchem as dc
from deepchem.data import DiskDataset, NumpyDataset
import arrow
import re
import errno
import optuna
from optuna.storages import RetryFailedTrialCallback
import json
import pickle as pkl
from itertools import product
from datetime import date
import torch
import torch.nn.functional as F
import shutil
import logging
import warnings

warnings.filterwarnings("ignore")

# ------------ Log -------------

SAVER_LOGGER = logging.getLogger("SynProtX.Saver")
MISC_LOGGER = logging.getLogger("SynProtX.MISC")
SYS_LOGGER = logging.getLogger("SYSTEM")

SAVER_LOGGER.setLevel(5)
SYS_LOGGER.setLevel(5)

# ------------ CUDA Setup -------------

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# torch.cuda.set_device(1)

torch.backends.cudnn.benchmark = True

# ------------ Confidence Interval -------------


def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return round(m, 4), round(pm, 4)


# ------------ Metrics for Classification -------------


class AUCPR(object):
    def __init__(self):
        super(AUCPR, self).__init__()
        self.name = "AUCPR"

    def __call__(self, answer, label):
        assert len(answer) == len(label)
        label = torch.Tensor(label)
        answer = torch.Tensor(answer)
        (
            precision,
            recall,
            _,
        ) = binary_precision_recall_curve(answer, label)
        aucpr_all = -np.trapz(precision, recall)
        return round(aucpr_all.tolist(), 4)


class Accuracy(object):
    def __init__(self):
        super(Accuracy, self).__init__()
        self.name = "Accuracy"

    def __call__(self, answer, label):
        assert len(answer) == len(label)
        label = torch.Tensor(label)
        answer = torch.Tensor(answer)
        accuracy = binary_accuracy(answer, label)
        return round(accuracy.tolist(), 4)


class Balanced_Accuracy(object):
    def __init__(self):
        super(Balanced_Accuracy, self).__init__()
        self.name = "Balanced_Accuracy"

    def __call__(self, answer, label):
        assert len(answer) == len(label)
        label = torch.Tensor(label)
        answer = torch.Tensor(answer)
        recall = binary_recall(answer, label)
        specificity = binary_specificity(answer, label)
        bal_accuracy = (recall + specificity) / 2
        return round(bal_accuracy.tolist(), 4)


class AUROC(object):
    def __init__(self):
        super(AUROC, self).__init__()
        self.name = "AUROC"

    def __call__(self, answer, label):
        assert len(answer) == len(label)
        label = torch.Tensor(label)
        answer = torch.Tensor(answer)
        auroc = binary_auroc(answer, label)
        return round(auroc.tolist(), 4)


class MCC(object):
    def __init__(self):
        super(MCC, self).__init__()
        self.name = "MCC"

    def __call__(self, answer, label):
        assert len(answer) == len(label)
        label = torch.Tensor(label)
        answer = torch.Tensor(answer)
        mcc = binary_matthews_corrcoef(answer, label)
        return round(mcc.tolist(), 4)


class Kappa(object):
    def __init__(self):
        super(Kappa, self).__init__()
        self.name = "Kappa"

    def __call__(self, answer, label):
        assert len(answer) == len(label)
        label = torch.Tensor(label)
        answer = torch.Tensor(answer)
        kappa = binary_cohen_kappa(answer, label)
        return round(kappa.tolist(), 4)


class BCE(object):
    def __init__(self):
        super(BCE, self).__init__()
        self.name = "BCE"

    def __call__(self, answer, label):
        assert len(answer) == len(label)
        label = torch.Tensor(label)
        answer = torch.Tensor(answer)
        bce = F.binary_cross_entropy(answer, label, reduction="mean")
        return round(bce.tolist(), 4)


class F1(object):
    def __init__(self):
        super(F1, self).__init__()
        self.name = "F1"

    def __call__(self, answer, label):
        assert len(answer) == len(label)
        label = torch.Tensor(label)
        answer = torch.Tensor(answer)
        f1 = binary_f1_score(answer, label)
        return round(f1.tolist(), 4)


class Precision(object):
    def __init__(self):
        super(Precision, self).__init__()
        self.name = "Precision"

    def __call__(self, answer, label):
        assert len(answer) == len(label)
        label = torch.Tensor(label)
        answer = torch.Tensor(answer)
        precision = binary_precision(answer, label)
        return round(precision.tolist(), 4)


class Recall(object):
    def __init__(self):
        super(Recall, self).__init__()
        self.name = "Recall"

    def __call__(self, answer, label):
        assert len(answer) == len(label)
        label = torch.Tensor(label)
        answer = torch.Tensor(answer)
        recall = binary_recall(answer, label)
        return round(recall.tolist(), 4)


class Specificity(object):
    def __init__(self):
        super(Specificity, self).__init__()
        self.name = "Specificity"

    def __call__(self, answer, label):
        assert len(answer) == len(label)
        label = torch.Tensor(label)
        answer = torch.Tensor(answer)
        specificity = binary_specificity(answer, label)
        return round(specificity.tolist(), 4)


# ------------ Metrics for Regression -------------


class RMSE(object):
    def __init__(self):
        super(RMSE, self).__init__()
        self.name = "RMSE"

    def __call__(self, answer, label):
        assert len(answer) == len(label)
        label = torch.Tensor(label)
        answer = torch.Tensor(answer)
        rmse = np.sqrt(mean_squared_error(answer, label))
        return round(rmse.tolist(), 4)


class MAE(object):
    def __init__(self):
        super(MAE, self).__init__()
        self.name = "MAE"

    def __call__(self, answer, label):
        assert len(answer) == len(label)
        label = torch.Tensor(label)
        answer = torch.Tensor(answer)
        mae = mean_absolute_error(answer, label)
        return round(mae.tolist(), 4)


class MSE(object):
    def __init__(self):
        super(MSE, self).__init__()
        self.name = "MSE"

    def __call__(self, answer, label):
        assert len(answer) == len(label)
        label = torch.Tensor(label)
        answer = torch.Tensor(answer)
        mse = mean_squared_error(answer, label)
        return round(mse.tolist(), 4)


class PCC(object):
    def __init__(self):
        super(PCC, self).__init__()
        self.name = "PCC"

    def __call__(self, answer, label):
        assert len(answer) == len(label)
        label = torch.Tensor(label)
        answer = torch.Tensor(answer)
        pcc = pearson_corrcoef(answer, label)
        return round(pcc.tolist(), 4)


class R2(object):
    def __init__(self):
        super(R2, self).__init__()
        self.name = "R2"

    def __call__(self, answer, label):
        assert len(answer) == len(label)
        label = torch.Tensor(label)
        answer = torch.Tensor(answer)
        r_squared = r2_score(answer, label)
        return round(r_squared.tolist(), 4)


class SCC(object):
    def __init__(self):
        super(SCC, self).__init__()
        self.name = "SCC"

    def __call__(self, answer, label):
        assert len(answer) == len(label)
        label = torch.Tensor(label)
        answer = torch.Tensor(answer)
        srcc = spearman_corrcoef(answer, label)
        return round(srcc.tolist(), 4)


# ------------ Random Seed -------------


def set_seed(new_seed=None):
    if new_seed is None:
        new_seed = random.randrange(1000)
    torch.manual_seed(new_seed)
    np.random.seed(new_seed)
    return new_seed


def set_split_seed(new_seed=None):
    if new_seed is None:
        new_seed = random.randrange(1000)

    return new_seed


# -------------- Utils ---------------


def get_original_seed(repeat, RUN_DIR):
    listdir_round = os.listdir(RUN_DIR)
    for dir_round in listdir_round:
        if f"repeat_{repeat}" in dir_round:
            split_seed = int(dir_round.split("_")[-1])
            return split_seed
    raise RuntimeError("split_seed not found")


def load_pickle(file_name):
    with open(file_name, "rb") as f:
        return pkl.load(f)


def load_tensor(file_name, dtype):
    return [dtype(d).to(DEVICE) for d in np.load(file_name)]


def load_tensor_cpu(file_name, dtype):
    return [dtype(d).to(torch.device("cpu")) for d in np.load(file_name)]


def customslice(data_list: list, index: list):
    new_data = []
    exceed_limit_count = 0
    for i in index:
        if i < len(data_list):
            new_data.append(data_list[i])
        else:
            exceed_limit_count += 1
    if exceed_limit_count > 0:
        MISC_LOGGER.warn(
            f"Exceed limit count {exceed_limit_count} from list of size {len(data_list)}. Omitting these indices."
        )
    return new_data


def load_dataset(adj, smiles, Y, ID):
    d = []
    data_dict = pd.read_pickle("./data/feature_dicts/MTL_ALL_WT_MT.pkl")

    print("Load dataset No.(label,smiles):", len(Y), len(smiles))

    atom_feat = np.array([data_dict["smiles_to_atom_info"][x] for x in smiles])
    atom_idx = np.array(
        [data_dict["smiles_to_atom_neighbors"][x] for x in smiles], dtype=int
    )
    edge_feat = np.array([data_dict["smiles_to_bond_info"][x] for x in smiles])
    edge_idx = np.array(
        [data_dict["smiles_to_bond_neighbors"][x] for x in smiles], dtype=int
    )

    for idx, i in enumerate(ID):
        x = torch.Tensor(atom_feat[i])
        edge_index = torch.LongTensor(edge_idx[i].T)
        edge_attr = torch.LongTensor(edge_feat[i])

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=Y[idx],
            adj=adj[idx],
            ID=i,
        )
        d.append(data)

    return d


def isnan(x):
    """Simple utility to see what is NaN"""
    return x != x


class StatusReport(object):
    def __init__(
        self,
        hyperpath,
        database,
        hypertune_stop_flag=False,
        repeat=0,
        fold=0,
        epoch=0,
        run_dir=None,
    ):
        self._run_dir = run_dir
        self._status = {
            "hypertune_stop_flag": hypertune_stop_flag,
            "repeat": repeat,
            "fold": fold,
            "epoch": epoch,
            "hyperpath": hyperpath,  # YYYY-MM-DD_HyperRunNo.
            "database": database,
        }

    def set_run_dir(self, run_dir):
        self._run_dir = run_dir
        with open(f"{self._run_dir}/status.json", "w") as status_file:
            json.dump(self._status, status_file, indent=4)

    @classmethod
    def resume_run(cls, run_dir):
        with open(f"{run_dir}/status.json", "r") as status_file:
            status = json.load(status_file)
        return cls(
            status["hyperpath"],
            status["database"],
            status["hypertune_stop_flag"],
            status["repeat"],
            status["fold"],
            status["epoch"],
            run_dir=run_dir,
        )

    def update(self, data):
        assert all(key in self._status.keys() for key in data)
        self._status.update(data)
        with open(f"{self._run_dir}/status.json", "w") as status_file:
            json.dump(self._status, status_file, indent=4)

    def get_status(self):
        return self._status.values()

    def __call__(self):
        return self._status


def calculate_score(pred, label, criterion, num_tasks):
    n_tasks = num_tasks
    squeeze_total_pred = np.array([])
    squeeze_total_label = np.array([])
    score_list = []
    for i in range(pred.shape[1]):
        task_label = label[:, i]
        task_pred = pred[:, i]
        task_masks = ~np.isnan(task_label)
        masked_label = np.atleast_1d(np.squeeze(task_label[task_masks]))
        masked_pred = np.atleast_1d(np.squeeze(task_pred[task_masks]))
        # print(squeeze_total_label)
        # print(masked_pred)
        squeeze_total_label = np.concatenate((squeeze_total_label, masked_label))
        squeeze_total_pred = np.concatenate((squeeze_total_pred, masked_pred))
        score_list.append(float(criterion(masked_pred, masked_label)))
    if n_tasks > 1:
        score_list.insert(0, float(criterion(squeeze_total_pred, squeeze_total_label)))
    return score_list


class ResultsReport(object):
    def __init__(self, target, metrics, run_dir: str, num_tasks, REPEATS, FOLDS):
        assert all(param is not None for param in (target, metrics))
        self._metrics = metrics
        self._metrics_name = [criterion.name for criterion in metrics]
        self._target = target
        self._run_dir = run_dir
        self._n_tasks = num_tasks
        self._resultsdf_col = [i for i in target]
        if self._n_tasks > 1:
            self._resultsdf_col.insert(0, "Overall")
        results_summary_dir = f"{run_dir}/ResultSummary.csv"
        if os.path.exists(results_summary_dir):
            self._resultsdf = pd.read_csv(
                results_summary_dir, header=[0, 1], index_col=[0, 1]
            )
        else:
            index = pd.MultiIndex.from_product(
                [list(range(REPEATS)), list(range(FOLDS))]
            )
            index = index.set_names(["repeat", "fold"])
            self._resultsdf = pd.DataFrame(
                columns=pd.MultiIndex.from_product(
                    (self._resultsdf_col, self._metrics_name)
                ),
                index=index,
            )

    def report_score(self, test_pred, test_label, repeat, fold):
        for criterion in self._metrics:
            score = calculate_score(test_pred, test_label, criterion, self._n_tasks)

            # resultsdf.xs(criterion.name, axis=1, level=1, drop_level=False).iloc[fold] = score
            self._resultsdf.loc[
                (repeat, fold), pd.IndexSlice[:, criterion.name]
            ] = score

        self._resultsdf.to_csv(
            self._run_dir + "/ResultSummary.csv", float_format="%.4f"
        )
        return self._resultsdf

    def report_by_target(self):
        outtext_by_target = []
        for col in self._resultsdf.columns:
            mean, interval = compute_confidence_interval(self._resultsdf[col])
            outtext_by_target.extend((mean, interval))
        resultsdf_by_target = pd.DataFrame(
            columns=pd.MultiIndex.from_product(
                (self._resultsdf_col, self._metrics_name, ["mean", "interval"])
            ),
            index=[0],
        )
        resultsdf_by_target.iloc[0] = outtext_by_target
        resultsdf_by_target = resultsdf_by_target.stack(0).droplevel(
            0
        )  # .swaplevel(0,1)
        resultsdf_by_target = resultsdf_by_target.reindex(
            columns=product(self._metrics_name, ["mean", "interval"]),
            index=self._resultsdf_col,
        )
        resultsdf_by_target.to_csv(
            self._run_dir + "/ResultSummary_ByTarget.csv", float_format="%.4f"
        )
        return resultsdf_by_target

    def get_dataframe(self):
        return self._resultsdf


# -------------- Saver ---------------


class Saver(object):
    def __init__(self, round_dir, max_epoch):
        super(Saver, self).__init__()
        self.round_dir = round_dir
        if self.round_dir[-1] != "/":
            self.round_dir += "/"
        self.ckpt_dir = self.round_dir + "checkpoints/"
        self.results_dir = self.round_dir + "results/"

        self.ckpt_count = 0
        self.EarlyStopController = EarlyStopController()
        self.max_epoch = max_epoch

    def SaveModel(self, model, optim, repeat, fold, epoch, scores):
        # state = {'model': model, 'optimizer': optimizer, 'epoch': epoch}
        ckpt_name = os.path.join(self.ckpt_dir, f"epoch{epoch}.pt")
        if not os.path.exists(os.path.dirname(ckpt_name)):
            try:
                os.makedirs(os.path.dirname(ckpt_name))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        torch.save([model, optim], ckpt_name)

        result_file_name = self.results_dir + str(epoch) + ".json"
        if not os.path.exists(os.path.dirname(result_file_name)):
            try:
                os.makedirs(os.path.dirname(result_file_name))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        with open(result_file_name, "w") as f:
            json.dump(scores, f, indent=4)

        # with open(f'{self.round_dir}/traintestloss.csv','a') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow([scores['train_loss'],scores['test_loss']])

        SAVER_LOGGER.info(
            f"Repeat {repeat} : Fold {fold} : Epoch {epoch} - Model saved."
        )

        ShouldStop = self.EarlyStopController.ShouldStop(scores, self.ckpt_count)

        if ShouldStop:
            BestValue, BestModelCkpt = self.EarlyStopController.BestModel()
            print("Early stop.")
            print("The Best model's ckpt idx is: ", BestModelCkpt)
            print("The Best Valid Value is: ", BestValue)
            # delete other models
            self.DeleteUselessCkpt(BestModelCkpt)
            return True, BestModelCkpt, BestValue

        elif (
            self.ckpt_count == self.max_epoch - 1
        ):  # ckpt_count start from 0 while maxepoch is with respect to 1
            BestValue, BestModelCkpt = self.EarlyStopController.BestModel()
            print("The model didn't stop.")
            print("The Best model's ckpt idx is: ", BestModelCkpt)
            print("The Best Valid Value is: ", BestValue)
            self.DeleteUselessCkpt(BestModelCkpt)
            return False, BestModelCkpt, BestValue
        else:
            self.ckpt_count += 1
            BestValue, BestModelCkpt = self.EarlyStopController.BestModel()
            return False, BestModelCkpt, BestValue

    def DeleteUselessCkpt(self, BestModelCkpt):
        file_names = os.listdir(self.ckpt_dir)
        for file_name in file_names:
            ckpt_idx = int(re.findall("\d+", file_name)[-1])
            if int(ckpt_idx) != BestModelCkpt:
                exact_file_path = self.ckpt_dir + file_name
                os.remove(exact_file_path)

    def LoadModel(self, ckpt=None):
        dir_files = os.listdir(self.ckpt_dir)  # list of the checkpoint files
        if dir_files:
            dir_files = sorted(
                dir_files,
                key=lambda x: os.path.getctime(os.path.join(self.ckpt_dir, x)),
            )
            last_model_ckpt = dir_files[-1]  # find the latest checkpoint file.
            model, optim = torch.load(os.path.join(self.ckpt_dir, last_model_ckpt))
            current_epoch = int(re.findall("\d+", last_model_ckpt)[-1])
            self.ckpt_count = (
                current_epoch + 1
            )  # update the ckpt_count, get rid of overwriting the existed checkpoint files.
            return model, optim
        else:
            return None, None


# -------------- Early Stopping ---------------


class EarlyStopController(object):
    def __init__(self):
        super(EarlyStopController, self).__init__()
        self.MetricName = "valid_loss"
        self.MaxResult = 9e8
        self.MaxResultModelIdx = None
        self.LastResult = 0
        self.LowerThanMaxNum = 0
        self.DecreasingNum = 0
        self.LowerThanMaxLimit = 30
        self.DecreasingLimit = 10
        self.TestResult = []

    def ShouldStop(self, score, ckpt_idx):
        MainScore = score[self.MetricName]
        if self.MaxResult > MainScore:
            self.MaxResult = MainScore
            self.MaxResultModelIdx = ckpt_idx
            self.LowerThanMaxNum = 0
            self.DecreasingNum = 0
            # all set to 0.
        else:
            # decreasing, start to count.
            self.LowerThanMaxNum += 1
            if MainScore > self.LastResult:
                # decreasing consistently.
                self.DecreasingNum += 1
            else:
                self.DecreasingNum = 0
        self.LastResult = MainScore

        if self.LowerThanMaxNum > self.LowerThanMaxLimit:
            return True
        if self.DecreasingNum > self.DecreasingLimit:
            return True
        return False

    def BestModel(self):
        # print(self.MaxResultModelIdx)
        # print(self.TestResult)
        return self.MaxResult, self.MaxResultModelIdx


# -------------- Dataset ---------------


def create_kfold_dataset(dc_tvset, folds, splitter, seed, RUN_DIR):
    os.makedirs(f"{RUN_DIR}/.tmpfolder/", exist_ok=True)
    directories = []
    for fold in range(folds):
        directories.extend(
            [f"{RUN_DIR}/.tmpfolder/{fold}-train", f"{RUN_DIR}/.tmpfolder/{fold}-test"]
        )
    for fold, (dc_trainset, dc_validset) in enumerate(
        splitter.k_fold_split(dc_tvset, k=folds, seed=seed, directories=directories)
    ):
        pass


def get_kfold_dataset(fold, RUN_DIR):
    trainset = NumpyDataset.from_DiskDataset(
        DiskDataset(data_dir=f"{RUN_DIR}/.tmpfolder/{fold}-train")
    )
    testset = NumpyDataset.from_DiskDataset(
        DiskDataset(data_dir=f"{RUN_DIR}/.tmpfolder/{fold}-test")
    )

    return trainset, testset


def del_tmpfolder(RUN_DIR):
    shutil.rmtree(f"{RUN_DIR}/.tmpfolder/")


# -------------- Model Architecture ---------------


class SynProtX_predictor(nn.Module):
    def __init__(
        self,
        node_feat_size: int,
        edge_feat_size: int,
        out_channels: int,
        graph_feat_size: int,
        num_layers_attentive: int,
        num_timesteps: int,
        dropout_attentive_rate: float,
        num_layers_predictor: int,
        dropout_predictor_rate: float,
        gene_feat_size: int,
        n_tasks: int,
        regression: bool,
        args,
        dropout_gene: float,
        MLPu=[2048, 512],
    ):
        super().__init__()

        # Print hyperparameters
        print("---- Hyperparameters Setting ----")
        print("node_feat_size:", node_feat_size)
        print("edge_feat_size:", edge_feat_size)
        print("graph_feat_size:", graph_feat_size)
        print("out_channels:", out_channels)
        print("num_layers_attentive:", num_layers_attentive)
        print("num_timesteps:", num_timesteps)
        print("dropout_attentive_rate:", dropout_attentive_rate)
        print("num_layers_predictor:", num_layers_predictor)
        print("dropout_predictor_rate:", dropout_predictor_rate)
        print("gene_feat_size:", gene_feat_size)
        print("gene_feat_size:", gene_feat_size)
        print("MLP:", MLPu)

        # Args
        self.use_feamol = not args.no_feamol
        self.use_feagene = not args.no_feagene
        self.use_feaprot = not args.no_feaprot

        # AttentiveFP drug
        self.drug1 = AttentiveFP(
            node_feat_size,
            graph_feat_size,
            out_channels,
            edge_feat_size,
            num_layers_attentive,
            num_timesteps,
            dropout_attentive_rate,
        )
        self.drug2 = AttentiveFP(
            node_feat_size,
            graph_feat_size,
            out_channels,
            edge_feat_size,
            num_layers_attentive,
            num_timesteps,
            dropout_attentive_rate,
        )

        # DNN Gene expression
        self.reduction = nn.Sequential(
            nn.Linear(gene_feat_size, MLPu[0]),
            nn.ReLU(),
            nn.Dropout(dropout_gene),
            nn.Linear(MLPu[0], MLPu[1]),
            nn.ReLU(),
            nn.Dropout(dropout_gene),
            nn.Linear(MLPu[1], out_channels * 2),
            nn.ReLU(),
        )

        # DNN predictor
        self.num_layers_predictor = num_layers_predictor
        self.dropout_predictor = nn.Dropout(dropout_predictor_rate)
        self.predictor_fc1 = nn.Linear(
            (
                (self.use_feamol * out_channels * 2)
                + (self.use_feagene * out_channels * 2)
                + (self.use_feaprot * out_channels * 2)
            ),
            512,
        )
        self.predictor_fc2 = nn.Linear(512, 256)
        self.predictor_fc3 = nn.Linear(256, 128)
        self.out_predictor_fc2 = nn.Linear(512, n_tasks)
        self.out_predictor_fc3 = nn.Linear(256, n_tasks)
        self.out_predictor_fc4 = nn.Linear(128, n_tasks)

        # Activation FN
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()

        # Regression mode
        self.regression = regression

        # protein exp layers
        self.prot_exp_fc_1 = nn.Linear(6688, 1024)
        self.prot_exp_fc_2 = nn.Linear(1024, 512)
        self.prot_exp_fc_3 = nn.Linear(512, out_channels * 2)

        # Regression mode
        self.regression = regression

    def forward(self, data):
        # AttentiveFP drug 1
        node_features_1, edge_features_1, edge_index_1, batch = (
            data.x[0].to(DEVICE),
            data.edge_attr[0].to(DEVICE),
            data.edge_index[0].to(DEVICE),
            data.batch.to(DEVICE),
        )
        drug1_molecules = self.drug1(
            node_features_1,
            edge_index=edge_index_1,
            edge_attr=edge_features_1,
            batch=batch,
        )

        # AttentiveFP drug 2
        node_features_2, edge_features_2, edge_index_2, batch = (
            data.x[1].to(DEVICE),
            data.edge_attr[1].to(DEVICE),
            data.edge_index[1].to(DEVICE),
            data.batch.to(DEVICE),
        )
        drug2_molecules = self.drug2(
            node_features_2,
            edge_index=edge_index_2,
            edge_attr=edge_features_2,
            batch=batch,
        )

        cell = data.gene_exp.to(DEVICE)
        cell = F.normalize(cell, 2, 1)
        gene_exp = self.reduction(cell)

        # protein exp

        init_feat = data.prot_exp.to(DEVICE, dtype=torch.float32)
        prot_exp = torch.reshape(init_feat, (-1, 6688))
        prot_exp = self.prot_exp_fc_1(prot_exp)
        prot_exp = self.elu(prot_exp)
        prot_exp = self.prot_exp_fc_2(prot_exp)
        prot_exp = self.elu(prot_exp)
        prot_exp = self.prot_exp_fc_3(prot_exp)
        prot_exp = self.relu(prot_exp)

        # Concatenate
        y_final = torch.Tensor().to(DEVICE)
        if self.use_feamol:
            y_final = torch.cat((y_final, drug1_molecules), 1)
            y_final = torch.cat((y_final, drug2_molecules), 1)
        if self.use_feagene:
            y_final = torch.cat((y_final, gene_exp), 1)
        if self.use_feaprot:
            y_final = torch.cat((y_final, prot_exp), 1)

        # DNN predictor
        output = self._DNNPredictor(y_final)

        # Regression mode
        if not self.regression:
            output = torch.sigmoid(output)

        # print(output)

        return output

    def _DNNPredictor(self, y_final):
        if self.num_layers_predictor == 2:
            x_pred = self.predictor_fc1(y_final)
            x_pred = self.relu(x_pred)
            x_pred = self.dropout_predictor(x_pred)
            output = self.out_predictor_fc2(x_pred)

        elif self.num_layers_predictor == 3:
            x_pred = self.predictor_fc1(y_final)
            x_pred = self.relu(x_pred)
            x_pred = self.predictor_fc2(x_pred)
            x_pred = self.relu(x_pred)
            x_pred = self.dropout_predictor(x_pred)
            output = self.out_predictor_fc3(x_pred)

        elif self.num_layers_predictor == 4:
            x_pred = self.predictor_fc1(y_final)
            x_pred = self.relu(x_pred)
            x_pred = self.predictor_fc2(x_pred)
            x_pred = self.relu(x_pred)
            x_pred = self.predictor_fc3(x_pred)
            x_pred = self.relu(x_pred)
            x_pred = self.dropout_predictor(x_pred)
            output = self.out_predictor_fc4(x_pred)

        else:
            print("Error DNNPredictor")

        return output


class AttentiveFP_DGL(nn.Module):
    def __init__(
        self,
        node_feat_size,
        edge_feat_size,
        num_layers_attentive=2,
        num_timesteps=2,
        graph_feat_size=200,
        dropout_attentive_rate=0.0,
    ):
        super().__init__()

        self.gnn = AttentiveFPGNN(
            node_feat_size=node_feat_size,
            edge_feat_size=edge_feat_size,
            num_layers=num_layers_attentive,
            graph_feat_size=graph_feat_size,
            dropout=dropout_attentive_rate,
        )

        self.readout = AttentiveFPReadout(
            feat_size=graph_feat_size,
            num_timesteps=num_timesteps,
            dropout=dropout_attentive_rate,
        )

    def forward(self, g, node_feats, edge_feats, get_node_weight=False):
        node_feats = self.gnn(g, node_feats, edge_feats)
        return self.readout(g, node_feats, get_node_weight)


# -------------- Trainer & Tester ---------------


class ModelWrapper(object):
    def __init__(self, model, optimizer=None, num_tasks=1, regression_mode=False):
        self.model = model
        self.optimizer = optimizer
        self.n_tasks = num_tasks
        self.regression_mode = regression_mode

    def fit(self, train_loader):
        self.model.train()

        total_pred = torch.Tensor([]).reshape(-1, self.n_tasks).to(DEVICE)
        total_label = torch.Tensor([]).reshape(-1, self.n_tasks).to(DEVICE)

        for i, data in enumerate(train_loader):
            label = torch.cat([d.y for d in data]).to(DEVICE)
            label = torch.reshape(label, (-1, self.n_tasks)).to(DEVICE)
            self.optimizer.zero_grad()
            pred = self.model(data)
            loss = self.multiloss(pred, label, self.regression_mode)
            total_pred = torch.cat((total_pred, pred), axis=0)
            total_label = torch.cat((total_label, label), axis=0)
            loss.backward()
            self.optimizer.step()
            torch.cuda.empty_cache()
        return total_pred.detach().cpu().numpy(), total_label.detach().cpu().numpy()

    def predict(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            total_pred = torch.Tensor([]).reshape(-1, self.n_tasks).to(DEVICE)
            total_label = torch.Tensor([]).reshape(-1, self.n_tasks).to(DEVICE)

            for data in test_loader:
                label = torch.cat([d.y for d in data]).to(DEVICE)
                label = torch.reshape(label, (-1, self.n_tasks)).to(DEVICE)
                pred = self.model(data)
                total_pred = torch.cat((total_pred, pred), axis=0)
                total_label = torch.cat((total_label, label), axis=0)
                torch.cuda.empty_cache()

        return total_pred.detach().cpu().numpy(), total_label.detach().cpu().numpy()

    def multiloss(self, output_vec, target_vec, regression_mode):
        total_output = torch.Tensor([]).to(DEVICE)
        total_target = torch.Tensor([]).to(DEVICE)

        criterion = (
            torch.nn.MSELoss(reduction="mean")
            if regression_mode
            else torch.nn.BCELoss(reduction="mean")
        )

        for x in range(len(target_vec)):
            masks = ~isnan(target_vec[x])

            if target_vec[x][masks].nelement() == 0:
                loss = [torch.sqrt(torch.tensor(1e-20)), torch.tensor(0.0)]
                continue
            else:  # non nans
                task_output_vec = output_vec[x][masks]
                task_target_vec = target_vec[x][masks]
                total_output = torch.cat((total_output, task_output_vec))
                total_target = torch.cat((total_target, task_target_vec))

            overall_loss = criterion(total_output, total_target)

        return overall_loss


# -------------- Hyperparameter Tuning ---------------


def get_best_trial(hyperrun_dir):
    with open(hyperrun_dir + "/besthyperparam.json", "r") as jsonfile:
        best_trial_param = json.load(jsonfile)
    return best_trial_param


def model_tuning(
    trial,
    tvset,
    dc_tvset,
    resume_flag=False,
    node_feat_size=39,
    edge_feat_size=12,
    gene_feat_size=714,
    num_epoch=50,
    n_tasks=1,
    regression=False,
    arguments=None,
    RUNDIR=None,
    FOLDS=None,
):
    node_dimension = node_feat_size
    edge_dimension = edge_feat_size
    gene_dimension = gene_feat_size
    num_tasks = n_tasks
    regression_mode = regression
    args = arguments
    RUN_DIR = RUNDIR

    num_layers_attentive = trial.suggest_int("num_layers_attentive", 1, 3)
    num_timesteps = trial.suggest_int("num_timesteps", 1, 3)
    dropout_attentive_rate = trial.suggest_float("dropout_attentive_rate", 0.1, 0.9)
    batchsize = trial.suggest_categorical("batchsize", [256])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    fingerprint_dim = trial.suggest_int("fingerprint_dim", 100, 300, step=25)
    num_layers_predictor = trial.suggest_int("num_layers_predictor", 2, 4)
    dropout_predictor_rate = trial.suggest_float("dropout_predictor_rate", 0.1, 0.9)
    output_units_num = trial.suggest_int("output_units_num", 25, 300, step=25)
    dropout_gene = trial.suggest_float("dropout_gene", 0.1, 0.9)
    MLPu1 = trial.suggest_categorical("MLPu1", [1024, 2048, 4096])
    MLPu2 = trial.suggest_categorical("MLPu2", [512, 1024])

    MLPu = [MLPu1, MLPu2]

    splitter = RandomSplitter()
    scores = []

    if not resume_flag:
        create_kfold_dataset(
            dc_tvset, folds=FOLDS, seed=42, splitter=splitter, RUN_DIR=RUN_DIR
        )

    for fold in range(FOLDS):
        print("====================")
        print(f"Fold:{fold+1}/{FOLDS}")
        print("====================")

        model = SynProtX_predictor(
            node_feat_size=node_dimension,
            edge_feat_size=edge_dimension,
            graph_feat_size=fingerprint_dim,
            out_channels=output_units_num,
            num_layers_attentive=num_layers_attentive,
            num_timesteps=num_timesteps,
            dropout_attentive_rate=dropout_attentive_rate,
            num_layers_predictor=num_layers_predictor,
            dropout_predictor_rate=dropout_predictor_rate,
            gene_feat_size=gene_dimension,
            n_tasks=num_tasks,
            regression=regression_mode,
            args=args,
            dropout_gene=dropout_gene,
            MLPu=MLPu,
        )
        model = DataParallel(model)
        model = model.to(DEVICE)

        # print('---- SynProtX Model Architecture ----')

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        wrapper = ModelWrapper(model, optimizer, num_tasks, regression_mode)

        dc_trainset, dc_validset = get_kfold_dataset(fold, RUN_DIR)

        trainset = customslice(tvset, dc_trainset.ids.tolist())
        validset = customslice(tvset, dc_validset.ids.tolist())

        train_loader = DataListLoader(
            trainset,
            batch_size=batchsize,
            shuffle=True,
            worker_init_fn=np.random.seed(0),
            num_workers=0,
            drop_last=True,
        )
        valid_loader = DataListLoader(
            validset,
            batch_size=1,
            shuffle=False,
            worker_init_fn=np.random.seed(0),
            num_workers=0,
        )

        best_loss = 200
        count = 0

        for epoch in range(num_epoch):
            # if epoch % decay_interval == 0:
            # wrapper.optimizer.param_groups[0]['lr'] *= lr_decay

            wrapper.fit(train_loader)
            valid_pred, valid_label = wrapper.predict(valid_loader)
            if not regression_mode:
                valid_loss = calculate_score(valid_pred, valid_label, BCE(), num_tasks)
            if regression_mode == True:
                valid_loss = calculate_score(valid_pred, valid_label, MSE(), num_tasks)

            print("valid_loss", valid_loss)

            if not regression_mode:
                loss_temp_CPR = calculate_score(
                    valid_pred, valid_label, AUCPR(), num_tasks
                )
                loss_temp_ROC = calculate_score(
                    valid_pred, valid_label, AUROC(), num_tasks
                )
                print("CPR:", loss_temp_CPR, " ROC:", loss_temp_ROC)

            if regression_mode == True:
                loss_temp_MAE = calculate_score(
                    valid_pred, valid_label, MAE(), num_tasks
                )
                loss_temp_RMSE = calculate_score(
                    valid_pred, valid_label, RMSE(), num_tasks
                )
                loss_temp_PCC = calculate_score(
                    valid_pred, valid_label, PCC(), num_tasks
                )
                print(
                    "MAE:",
                    loss_temp_MAE,
                    " RMSE:",
                    loss_temp_RMSE,
                    " PCC:",
                    loss_temp_PCC,
                )

            if best_loss > valid_loss[0]:
                best_loss = valid_loss[0]
                count = 0
            else:
                count += 1

            if count == 20:
                break

        torch.cuda.empty_cache()
        scores.append(best_loss)

    mean_loss = round(np.mean(scores), 4)

    print("mean_valid_loss", mean_loss)

    del trainset, validset, dc_trainset, dc_validset
    gc.collect()

    return mean_loss


def run_hyper_study(study_func, max_trials, study_name, hyperrun_dir):
    print("Start run hypertune...")

    storage = optuna.storages.RDBStorage(
        f"sqlite:///{hyperrun_dir}/hyperparameter-search.db",
        heartbeat_interval=1,
        failed_trial_callback=RetryFailedTrialCallback(),
        engine_kwargs={"connect_args": {"timeout": 100}},
    )

    study = optuna.create_study(
        storage=storage,
        direction="minimize",
        study_name=study_name,
        load_if_exists=True,
    )
    n_trials = max_trials - len(study.trials_dataframe())
    if n_trials != 0:
        study.optimize(study_func, n_trials=n_trials)
    trial = study.best_trial
    best_trial_param = dict()
    for key, value in trial.params.items():
        best_trial_param[key] = value
    with open(hyperrun_dir + "/besthyperparam.json", "w") as jsonfile:
        json.dump(best_trial_param, jsonfile, indent=4)

    return study


# -------------- Main ---------------


def main():
    start_time = arrow.now()
    start_time_formatted = start_time.format("DD/MM/YYYY HH:mm:ss")
    print("Start time:", start_time_formatted)

    database_path_dict = {
        r"almanac-breast": "./data/nps_intersected/ALMANAC/breast",
        r"almanac-lung": "./data/nps_intersected/ALMANAC/lung",
        r"almanac-ovary": "./data/nps_intersected/ALMANAC/ovary",
        r"almanac-skin": "./data/nps_intersected/ALMANAC/skin",
        r"friedman": "./data/nps_intersected/FRIEDMAN",
    }
    all_database_list = [i for i in database_path_dict.keys()]

    parser = argparse.ArgumentParser(
        description="Model configuration for SynProtX Model"
    )
    parser.add_argument(
        "-d",
        "--database",
        required=True,
        dest="database",
        help="Select database",
        choices=all_database_list,
    )
    parser.add_argument(
        "-m",
        "--mode",
        required=True,
        dest="mode",
        help="classification task or regression task",
        choices=["clas", "regr"],
    )
    parser.add_argument(
        "--load-hyper",
        required=False,
        dest="hyperpath",
        help="enter hyperparameter directory. if none, use new hyperparameters",
    )
    parser.add_argument(
        "--resume", dest="resume_run_dir", default=None, help="if exist, resume run"
    )
    parser.add_argument(
        "--no-feamol",
        dest="no_feamol",
        default=False,
        action="store_true",
        help="if added, disable Attentive FP for molecule",
    )
    parser.add_argument(
        "--no-feagene",
        dest="no_feagene",
        default=False,
        action="store_true",
        help="if added, disable DNN for gene expression",
    )
    parser.add_argument(
        "--no-feaprot",
        dest="no_feaprot",
        default=False,
        action="store_true",
        help="if added, disable GNN for protein expression",
    )
    parser.add_argument(
        "-x",
        "--debug",
        default=False,
        action="store_true",
        dest="debug",
        help="debug file/test run",
    )
    parser.add_argument("--log", dest="log_dir")

    args = parser.parse_args()

    modelname = r"SynProtX-AttFP"
    target = ["Loewe_score"]
    random.seed = 42

    if args.mode == "clas":
        mode = f"classification"
        regression_mode = False
        print("====", modelname, "- Classification Task ====")
    elif args.mode == "regr":
        mode = "regr"
        regression_mode = True
        print("====", modelname, "- Regression Task ====")
    else:
        raise RuntimeError(f"Invalid mode: {args.mode}")

    database = args.database
    print("Database:", database)

    hyperpath = args.hyperpath
    hypertune_stop_flag = args.hyperpath is not None

    testsplitname = r"random"
    hypersplitname = r"random"

    model_dir = os.path.join("results", modelname, database, mode)
    split_dir = os.path.join(model_dir, testsplitname)
    os.makedirs(split_dir, exist_ok=True)
    hyperparamsplit_dir = os.path.join(model_dir, "hyperparameters", hypersplitname)

    if args.resume_run_dir is None:
        resume_flag = False
        hyperpath = args.hyperpath
        database = args.database

        # Init new status
        status = StatusReport(hyperpath, database, hypertune_stop_flag)
        (
            hypertune_stop_flag,
            ck_repeat,
            ck_fold,
            ck_epoch,
            hyperpath,
            database,
        ) = status.get_status()
    else:
        original_run = f"{split_dir}/{args.resume_run_dir}"
        print(f"Reload experiment: {original_run}")
        SYS_LOGGER.info(f"RESUME RUN: {original_run}")
        resume_flag = True
        status = StatusReport.resume_run(original_run)
        (
            hypertune_stop_flag,
            ck_repeat,
            ck_fold,
            ck_epoch,
            hyperpath,
            database,
        ) = status.get_status()

    database_path = database_path_dict[database]

    if args.debug:
        print("DEBUG MODE")
        max_trials = 1
        max_tuning_epoch = 10
        REPEATS = 1
        FOLDS = 2
        max_epoch = 10
        printrate = 1
    else:
        print("NORMAL MODE")
        max_trials = 30
        max_tuning_epoch = 150
        REPEATS = 3
        FOLDS = 5
        max_epoch = 200
        printrate = 20

    num_tasks = len(target)
    print("No.task:", num_tasks)

    splitter = RandomSplitter()

    metrics = (
        [MSE(), RMSE(), MAE(), R2(), PCC(), SCC()]
        if regression_mode
        else [
            AUCPR(),
            AUROC(),
            F1(),
            MCC(),
            Kappa(),
            Accuracy(),
            Balanced_Accuracy(),
            Specificity(),
            Precision(),
            Recall(),
        ]
    )

    outtext_list = [database]

    if not args.no_feamol:
        print("Use Attentive FP")
        outtext_list.append(1)
    else:
        print("No Attentive FP")
        outtext_list.append(0)
    if not args.no_feagene:
        print("Use DNN Gene Expression")
        outtext_list.append(1)
    else:
        print("No DNN Gene Expression")
        outtext_list.append(0)
    if not args.no_feaprot:
        print("Use Protein Expression")
        outtext_list.append(1)
    else:
        print("No Protein Expression")
        outtext_list.append(0)

    # ------- Load Feature -------

    print("---- Feature Loading ----")

    np_label = np.load(f"{database_path}/label_cla_row.npy")
    drugcomb_raw = pd.read_pickle("./data/data_drugcomb.pkl")
    smiles_row = drugcomb_raw[f"molecule_structures_row"].to_numpy()
    smiles_col = drugcomb_raw[f"molecule_structures_col"].to_numpy()
    print("Label shape:", np_label.shape)

    # adj drug_row
    adj_row = load_tensor_cpu(f"{database_path}/adj_row.npy", torch.FloatTensor)
    for i in range(len(adj_row)):
        adj_row[i] = adj_row[i].unsqueeze(0)
    print("Done adj drug_row")

    # adj drug_col
    adj_col = load_tensor_cpu(f"{database_path}/adj_col.npy", torch.FloatTensor)
    for i in range(len(adj_col)):
        adj_col[i] = adj_col[i].unsqueeze(0)
    print("Done adj drug_col")

    # label
    Label = load_tensor_cpu(
        f"{database_path}/label_reg_row.npy"
        if regression_mode
        else f"{database_path}/label_cla_row.npy",
        torch.FloatTensor,
    )
    print("Done label")

    # ID
    ID = np.load(f"{database_path}/idx_row.npy")
    print("Done ID")

    # gene_exp
    gene = pd.read_pickle("./data/data_preprocessing_gene.pkl")
    gene = gene.loc[ID]
    gene = gene.reset_index(drop=True)
    gene_dimension = gene.shape[-1] - 2
    print("No.gene size:", gene_dimension)
    print("Done gene")

    # protein_exp (protein exp)
    prot_exp_node_feat = pd.read_pickle("./data/data_preprocessing_protein.pkl")
    columns = prot_exp_node_feat.columns
    prot_exp_node_feat.columns = ["id", "cell_line_name"] + [
        x.split(";")[1].split("_")[0] for x in columns[2:]
    ]
    prot_exp_node_feat = prot_exp_node_feat.loc[ID]
    prot_exp_node_feat = prot_exp_node_feat.reset_index(drop=True)
    prot_exp_node_feat = prot_exp_node_feat[prot_exp_node_feat.columns[:-4]]

    print("Done protein exp")

    # load drug_col
    dataset_col = load_dataset(adj_col, smiles_col, Label, ID)
    print("No.dataset drug_col:", len(dataset_col))
    print("Done dataset drug_col")

    del adj_col
    gc.collect()
    torch.cuda.empty_cache()

    # load drug_row
    dataset_row = load_dataset(adj_row, smiles_row, Label, ID)
    print("no.dataset drug_row:", len(dataset_row))
    print("done dataset drug_row")

    del adj_row

    gc.collect()
    torch.cuda.empty_cache()

    node_dimension = 39
    edge_dimension = 12

    # ------- Complie to Tensor -------

    print("---- Tensor Data Compliing ----")

    dataset_temp = []
    for i in range(len(np_label)):
        x = [dataset_row[i].x, dataset_col[i].x]
        edge_attr = [dataset_row[i].edge_attr, dataset_col[i].edge_attr]
        edge_index = [dataset_row[i].edge_index, dataset_col[i].edge_index]
        adj = [dataset_row[i].adj, dataset_col[i].adj]
        gene_exp = torch.Tensor([np.array(gene.loc[i].values[2:], dtype=float)])
        prot_exp = torch.Tensor(
            np.array(
                [[0 if x != x else x] for x in prot_exp_node_feat.loc[i].values[2:]]
            )
        )
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=dataset_row[i].y,
            adj=adj,
            ID=dataset_row[i].ID,
            num_nodes=x[0].shape[0],
            gene_exp=gene_exp,
            prot_exp=prot_exp,
        )
        dataset_temp.append(data)

    # ------- Complie to DeepChem -------

    print("---- DeepChem Data Compliing ----")

    if args.debug:
        n_data = 5000
        dataset_temp = dataset_temp[0:n_data]
        dc_dataset = dc.data.NumpyDataset(
            X=list(range(n_data)), y=np_label[0:n_data], ids=list(range(n_data))
        )
    else:
        dc_dataset = dc.data.NumpyDataset(
            X=list(range(len(np_label))), y=np_label, ids=list(range(len(np_label)))
        )

    # ------- Split Data -------

    print("---- Data Spliting ----")

    # data splitting
    dc_tvset, dc_testset = splitter.train_test_split(
        dc_dataset, frac_train=0.8, seed=42
    )

    print("No.dc_tvset:", len(dc_tvset))
    print("No.dc_testset:", len(dc_testset))

    tvset = customslice(dataset_temp, dc_tvset.ids.tolist())
    testset = customslice(dataset_temp, dc_testset.ids.tolist())

    dc_tvset = dc.data.NumpyDataset(
        X=dc_tvset.y, y=dc_tvset.y, ids=list(range(len(tvset)))
    )  # reindex

    test_loader = DataListLoader(testset, batch_size=1, shuffle=False, num_workers=0)

    del dc_testset
    gc.collect()

    print("---- Done! ----")

    # ------- Hyperparameters Optimization -------

    today_date = date.today().strftime("%Y-%m-%d")

    if not hypertune_stop_flag:
        print("---- Hyperparameter Optimization ----")

        if not resume_flag:
            num_hyperrun = 1
            hyperpath = f"{today_date}_HyperRun{num_hyperrun}"
            hyperrun_dir = os.path.join(hyperparamsplit_dir, hyperpath)
            while os.path.exists(hyperrun_dir):
                num_hyperrun += 1
                hyperpath = f"{today_date}_HyperRun{num_hyperrun}"
                hyperrun_dir = os.path.join(hyperparamsplit_dir, hyperpath)
            os.makedirs(hyperrun_dir)
            RUN_DIR = f"{split_dir}/{today_date}_HyperRun{num_hyperrun}-{hypersplitname}_TestRun1"
            os.makedirs(RUN_DIR)
            print("Create new experiment...")
            status.set_run_dir(RUN_DIR)
            status.update({"hyperpath": hyperpath})
        else:
            RUN_DIR = original_run
        print(f'Your run directory is "{RUN_DIR}"')
        log_dir = args.log_dir if args.log_dir is not None else RUN_DIR
        logging.basicConfig(level=5, filename=f"{log_dir}/.log", filemode="a")
        MISC_LOGGER.info(f"testset size = {len(testset)}")
        MISC_LOGGER.info(f"tvset size = {len(tvset)}")
        print("Start hyperparameters optimization...")

        def model_tuning_simplified(trial):
            return model_tuning(
                trial=trial,
                tvset=tvset,
                dc_tvset=dc_tvset,
                resume_flag=False,
                node_feat_size=node_dimension,
                edge_feat_size=edge_dimension,
                gene_feat_size=gene_dimension,
                num_epoch=max_tuning_epoch,
                n_tasks=num_tasks,
                regression=regression_mode,
                arguments=args,
                RUNDIR=RUN_DIR,
                FOLDS=FOLDS,
            )

        study_name = f"{modelname}-{database}-{mode}"
        hyperrun_dir = os.path.join(hyperparamsplit_dir, hyperpath)
        run_hyper_study(
            study_func=model_tuning_simplified,
            max_trials=max_trials,
            study_name=study_name,
            hyperrun_dir=hyperrun_dir,
        )
        hyperparam = get_best_trial(hyperrun_dir)
        hypertune_stop_flag = True
        status.update({"hypertune_stop_flag": True})

    else:
        print("Loading hyperparameters...")
        if not resume_flag:
            num_run = 1
            RUN_DIR = f"{split_dir}/{hyperpath}-{hypersplitname}_TestRun{num_run}"
            while os.path.exists(RUN_DIR):
                num_run += 1
                RUN_DIR = f"{split_dir}/{hyperpath}-{hypersplitname}_TestRun{num_run}"
            os.makedirs(RUN_DIR)
            status.set_run_dir(RUN_DIR)
        else:
            RUN_DIR = original_run
        print(f'Your run directory is "{RUN_DIR}"')
        log_dir = args.log_dir if args.log_dir is not None else RUN_DIR
        logging.basicConfig(level=5, filename=f"{RUN_DIR}/.log", filemode="a")
        MISC_LOGGER.info(f"testset size = {len(testset)}")
        MISC_LOGGER.info(f"tvset size = {len(tvset)}")
        hyperrun_dir = os.path.join(hyperparamsplit_dir, hyperpath)
        hyperparam = get_best_trial(hyperrun_dir)

    num_layers_attentive = hyperparam.get("num_layers_attentive")
    num_timesteps = hyperparam.get("num_timesteps")
    dropout_attentive_rate = hyperparam.get("dropout_attentive_rate")
    batchsize = hyperparam.get("batchsize")
    lr = hyperparam.get("lr")
    weight_decay = hyperparam.get("weight_decay")
    fingerprint_dim = hyperparam.get("fingerprint_dim")
    num_layers_predictor = hyperparam.get("num_layers_predictor")
    dropout_predictor_rate = hyperparam.get("dropout_predictor_rate")
    output_units_num = hyperparam.get("output_units_num")
    MLPu1 = hyperparam.get("MLPu1")
    MLPu2 = hyperparam.get("MLPu2")
    dropout_gene = hyperparam.get("dropout_gene")

    MLPu = [MLPu1, MLPu2]

    print("---- Done! ----")

    with open(RUN_DIR + "/hyperparameters.json", "w") as jsonfile:
        json.dump(hyperparam, jsonfile, indent=4)
    outtext_list.insert(0, os.path.basename(RUN_DIR))

    metrics_name = [criterion.name for criterion in metrics]

    results_report = ResultsReport(
        target,
        metrics,
        run_dir=RUN_DIR,
        num_tasks=num_tasks,
        REPEATS=REPEATS,
        FOLDS=FOLDS,
    )

    # ------- Run Experiment -------

    for repeat in range(ck_repeat, REPEATS):
        print("====================")
        print(f"Repeat:{repeat+1}/{REPEATS}")

        if resume_flag:
            split_seed = get_original_seed(repeat)
        else:
            split_seed = set_split_seed()
        create_kfold_dataset(
            dc_tvset, folds=FOLDS, seed=split_seed, splitter=splitter, RUN_DIR=RUN_DIR
        )

        for fold in range(ck_fold, FOLDS):
            dc_trainset, dc_validset = get_kfold_dataset(fold, RUN_DIR)
            trainset = customslice(tvset, dc_trainset.ids.tolist())
            validset = customslice(tvset, dc_validset.ids.tolist())

            print("====================")
            print(f"Fold:{fold+1}/{FOLDS}")
            print("====================")
            seed = set_seed()

            round_dir = f"{RUN_DIR}/repeat_{repeat}-fold_{fold}-split_seed_{split_seed}"
            if os.path.exists(round_dir):
                shutil.rmtree(round_dir)  # reset training process on this round
            os.makedirs(round_dir)
            saver = Saver(round_dir, max_epoch=max_epoch)

            train_loader = DataListLoader(
                trainset,
                batch_size=batchsize,
                shuffle=True,
                worker_init_fn=np.random.seed(seed),
                num_workers=0,
                drop_last=True,
            )
            valid_loader = DataListLoader(
                validset, batch_size=1, shuffle=False, num_workers=0
            )

            model = SynProtX_predictor(
                node_feat_size=node_dimension,
                edge_feat_size=edge_dimension,
                graph_feat_size=fingerprint_dim,
                out_channels=output_units_num,
                num_layers_attentive=num_layers_attentive,
                num_timesteps=num_timesteps,
                dropout_attentive_rate=dropout_attentive_rate,
                num_layers_predictor=num_layers_predictor,
                dropout_predictor_rate=dropout_predictor_rate,
                gene_feat_size=gene_dimension,
                n_tasks=num_tasks,
                regression=regression_mode,
                args=args,
                dropout_gene=dropout_gene,
                MLPu=MLPu,
            )
            model = DataParallel(model)
            model = model.to(DEVICE)
            optimizer = optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )

            wrapper = ModelWrapper(model, optimizer, num_tasks, regression_mode)

            stop_flag = False

            loss_train_list = []
            loss_valid_list = []

            for epoch in range(max_epoch):
                if epoch % printrate == 0:
                    print(f"Epoch:{epoch+1}/{max_epoch}")
                status.update({"repeat": repeat, "fold": fold, "epoch": epoch})
                if stop_flag:
                    print("Early stop at epoch", epoch)
                    break

                wrapper.fit(train_loader)
                train_pred, train_label = wrapper.predict(train_loader)
                valid_pred, valid_label = wrapper.predict(valid_loader)

                if not regression_mode:
                    train_loss = calculate_score(
                        train_pred, train_label, BCE(), num_tasks
                    )
                    valid_loss = calculate_score(
                        valid_pred, valid_label, BCE(), num_tasks
                    )
                if regression_mode == True:
                    train_loss = calculate_score(
                        train_pred, train_label, MSE(), num_tasks
                    )
                    valid_loss = calculate_score(
                        valid_pred, valid_label, MSE(), num_tasks
                    )

                loss_train_list.append(train_loss)
                loss_valid_list.append(valid_loss)

                scores = {"train_loss": train_loss[0], "valid_loss": valid_loss[0]}

                stop_flag, *_ = saver.SaveModel(
                    wrapper.model, wrapper.optimizer, repeat, fold, epoch, scores
                )

                if epoch % 10 == 0:
                    torch.cuda.empty_cache()

            # final model assessnent

            bestmodel, *_ = saver.LoadModel()
            bestwrapper = ModelWrapper(bestmodel)
            train_pred, train_label = bestwrapper.predict(train_loader)
            test_pred, test_label = bestwrapper.predict(test_loader)

            pt_dict = {
                "train_pred": train_pred,
                "train_label": train_label,
                "test_pred": test_pred,
                "test_label": test_label,
            }

            pkl.dump(pt_dict, open(round_dir + "/pred_true.pkl", "wb"))

            results_report.report_score(test_pred, test_label, repeat, fold)

            loss_np = np.asarray([loss_train_list, loss_valid_list]).reshape(-1, 2)
            np.savetxt(round_dir + "/loss.csv", loss_np, delimiter=",", fmt="%.3f")
            gc.collect()

        ck_fold = 0
        resume_flag = False

    print("Writing output...")

    resultsdf = results_report.get_dataframe()

    for col in resultsdf.columns:
        mean, interval = compute_confidence_interval(resultsdf[col])
        outtext_list.extend((mean, interval))

    resultsdf_col = (
        results_report.report_by_target().index.to_list() if num_tasks > 1 else target
    )
    torch.cuda.empty_cache()
    del_tmpfolder(RUN_DIR)

    end_time = arrow.now()
    end_time_formatted = end_time.format("DD/MM/YYYY HH:mm:ss")
    print("Finish time:", end_time_formatted)
    elapsed_time = end_time - start_time

    summaryfile = split_dir + "/AllExperimentsSummary.csv"
    if not os.path.exists(summaryfile):
        X = pd.DataFrame(
            columns=pd.MultiIndex.from_product(
                (resultsdf_col, metrics_name, ["mean", "interval"])
            )
        )
        X.insert(0, "Molecule_Attentive FP", np.nan)
        X.insert(1, "Gene_Exp", np.nan)
        X.insert(2, "Prot_Exp", np.nan)
        X.insert(0, "database", np.nan)
        X.insert(0, "experiment_name", np.nan)
        X.insert(1, "start_time", np.nan)
        X.insert(2, "end_time", np.nan)
        X.insert(3, "elapsed_time", np.nan)
        X.to_csv(summaryfile, index=False, float_format="%.4f")
    with open(summaryfile, "a") as outfile:
        outtext_list.insert(1, start_time_formatted)
        outtext_list.insert(2, end_time_formatted)
        outtext_list.insert(3, str(elapsed_time).split(".")[0])
        output_writer = csv.writer(outfile, delimiter=",")
        output_writer.writerow(outtext_list)


if __name__ == "__main__":
    main()
