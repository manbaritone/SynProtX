# -----------------------------------------------------------------------------
# SynProtX Inference Utility
# -----------------------------------------------------------------------------
"""synprotx_inference.py

Predict anticancer-drug‑pair synergy classification or regression for a given
cell‑line using a trained **SynProtX‑GAT‑FP** checkpoint.

Examples
~~~~~~~~
CLI usage

:: bash

    python synprotx_inference.py --smi1 "CCOc1ccc2c(c1)N=C(N)N(c3ccc(Cl)cc3)S2" --smi2 "CN1CCC(CC1)Nc2nccc3c2ncn3C" \ 
    --dataset ALMANAC-Breast --cell_line MCF7 --task classification --thr 0.5

    ---- Hyperparameters Setting ----
    node_feat_size: 39
    out_channels: 75
    dropout_rate: 0.3149202140836747
    num_layers_predictor: 2
    dropout_predictor_rate: 0.7018169381239597
    dropout_fpdnn_rate: 0.6312080891089443
    gene_feat_size: 714
    nheads: 6
    
    Example output
    
    ----------------------------------------------------------------
    Dataset: ALMANAC-Breast
    Cell Line: MCF7
    Task: classification
    SMILES1: CCOc1ccc2c(c1)N=C(N)N(c3ccc(Cl)cc3)S2
    SMILES2: CN1CCC(CC1)Nc2nccc3c2ncn3C
    Predicted Value: 0.8123
    Interpretation: Synergism
    ----------------------------------------------------------------

Run ``python synprotx_inference.py -h`` for details.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Union

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from torch import Tensor
from torch_geometric.data import Batch, Data

from gatfp import SynProtXgatfp3_predictor

# ----------------------------------------------------------------------------
# 1.  Helper – clean legacy / DataParallel keys
# ----------------------------------------------------------------------------

def _clean_state_dict(raw_sd: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Strip the annoying ``module.`` prefix and rename *ppi* → *prot_exp*."""
    new_sd = {}
    for k, v in raw_sd.items():
        if k.startswith("module."):
            k = k[7:]
        if k.startswith("ppi_fc_"):
            k = k.replace("ppi_fc_", "prot_exp_fc_")
        new_sd[k] = v
    return new_sd

# ----------------------------------------------------------------------------
# quick helper – safe row → (1 × n_features) tensor
# ----------------------------------------------------------------------------
def _expr_row_as_tensor(df: pd.DataFrame, key: str, n_feat: int, device) -> torch.Tensor:
    """
    Return the FIRST row that matches *key* as a (1, n_feat) float32 tensor.
    Works whether ``df.loc[key]`` yields a Series or a DataFrame (duplicate keys).
    Truncates / pads so the length is always exactly *n_feat*.
    """
    row = df.loc[key]
    if isinstance(row, pd.DataFrame):          # duplicate index → DataFrame
        row = row.iloc[0]                      # keep the first occurrence
    vec = row.to_numpy(dtype=np.float32)[:n_feat]          # guarantee length
    if vec.shape[0] < n_feat:                                  # never happens for gene,
        vec = np.pad(vec, (0, n_feat - vec.shape[0]))          # but safety‑net
    return torch.from_numpy(vec).unsqueeze(0).to(device)      # (1, n_feat)

# ----------------------------------------------------------------------------
# 2.  SMILES → graph  (exactly as in the training notebooks)
# ----------------------------------------------------------------------------

ATOM_LIST = [
    "B","C","N","O","F","Si","P","S","Cl","As","Se","Br","Te","I","At","other",
]
HYBRIDISATION = {
    Chem.rdchem.HybridizationType.SP: 0,
    Chem.rdchem.HybridizationType.SP2: 1,
    Chem.rdchem.HybridizationType.SP3: 2,
    Chem.rdchem.HybridizationType.SP3D: 3,
    Chem.rdchem.HybridizationType.SP3D2: 4,
    "other": 5,
}
BOND_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
    "unknown",
]

# -- low‑level feature helpers ------------------------------------------------

def _one_hot(x: Union[str, int], allowable: List[Union[str, int]]) -> List[int]:
    if x not in allowable:
        x = allowable[-1]
    return [int(x == a) for a in allowable]

def _atom_features(atom: Chem.Atom) -> List[int]:
    return (
        _one_hot(atom.GetSymbol(), ATOM_LIST)
        + _one_hot(atom.GetHybridization(), list(HYBRIDISATION.keys()))
        + _one_hot(atom.GetDegree(), list(range(6)))
        + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons(), atom.GetIsAromatic()]
    )

def _bond_features(bond: Chem.Bond) -> List[int]:
    return (
        _one_hot(bond.GetBondType(), BOND_LIST)
        + [bond.GetIsConjugated(), bond.IsInRing()]
        + _one_hot(str(bond.GetStereo()), [
            "STEREONONE", "STEREOANY", "STEREOZ", "STEREOE",
        ])
    )

# -- graph builder ------------------------------------------------------------

def smiles_to_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")

    x = torch.tensor([_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    e_idx, e_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = _bond_features(bond)
        e_idx.extend([[i, j], [j, i]])
        e_attr.extend([feat, feat])

    edge_index = torch.tensor(e_idx, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(e_attr, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

class _GraphCache:
    """Lazy graph generator with memoisation."""

    def __init__(self):
        self._cache: Dict[str, Data] = {}

    def __call__(self, smiles: str) -> Data:
        if smiles not in self._cache:
            self._cache[smiles] = smiles_to_graph(smiles)
        return self._cache[smiles]

_graph_gen = _GraphCache()

# ----------------------------------------------------------------------------
# 3.  Fingerprint helpers
# ----------------------------------------------------------------------------

def _fp_resize(bits: np.ndarray, new_len: int) -> np.ndarray:
    if bits.shape[0] >= new_len:
        return bits[:new_len]
    return np.concatenate([bits, np.zeros(new_len - bits.shape[0], dtype=bits.dtype)])

def _smiles_to_ecfp6(smiles: str, n_bits: int) -> torch.Tensor:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Bad SMILES: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)
    arr = np.zeros((2048,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    arr = arr[:n_bits] if n_bits < 2048 else arr
    return torch.from_numpy(arr)

def _pad(x, target):
    if x.size(1) < target:
        pad = torch.zeros(x.size(0), target - x.size(1), dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=1)
    return x

# ----------------------------------------------------------------------------
# 4.  Main public API
# ----------------------------------------------------------------------------

class SynProtXInference:
    """Light‑weight, checkpoint‑aware predictor."""

    DATASETS = [
        "ALMANAC-Breast", "ALMANAC-Lung", "ALMANAC-Ovary", "ALMANAC-Skin",
        "FRIEDMAN", "ONEIL",
    ]

    def __init__(self, task: str = "regression", dataset: str = "ALMANAC-Breast", device: str = "cpu"):
        # --------------- sanity checks --------------------------------------
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
        if dataset not in self.DATASETS:
            raise KeyError(f"Unknown dataset: {dataset}")

        self.task = task
        self.dataset = dataset
        self.device = torch.device(device)

        # --------------- locate files --------------------------------------
        fn_base   = f"SynProtX-GATFP_{dataset}"
        ckpt_path = Path("state_dict") / task / f"{fn_base}.pt"
        cfg_path  = Path("hyperparams") / task / f"{fn_base}.json"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        if not cfg_path.exists():
            raise FileNotFoundError(f"Hyper‑parameters JSON not found: {cfg_path}")

        # --------------- peek at checkpoint to discover true feature sizes --
        raw_obj = torch.load(ckpt_path, map_location="cpu")
        raw_sd  = raw_obj[0].state_dict() if isinstance(raw_obj, (list, tuple)) else raw_obj
        clean_sd = _clean_state_dict(raw_sd)

        # deduce dimensions from weight shapes --------------------------------
        fp_layer_in  = clean_sd["molfp_lin_drr.0.weight"].shape[1]
        with cfg_path.open() as fh:
            hp = json.load(fh)
        out_channels = hp["output_units_num"]
        # true ECFP‑6 length used during training
        self.fp_len = fp_layer_in - out_channels
        self.gene_len = clean_sd["reduction.0.weight"].shape[1]
        self.prot_len = clean_sd["prot_exp_fc_1.weight"].shape[1]
        self.atom_dim = clean_sd["drug1_gat1.lin_src.weight"].shape[1]

        # --------------- expression tables ----------------------------------
        self._load_expression_tables()

        # --------------- build model ----------------------------------------
        with cfg_path.open() as fh:
            hp = json.load(fh)

        args_stub = SimpleNamespace(
            no_feamol=False, no_feagene=False, no_feaprot=False, no_fp=False,
            drc_input_size=self.fp_len, drr_input_size=self.fp_len,
        )
        
        cfg = dict(
            node_feat_size=self.atom_dim,
            out_channels=hp.get("output_units_num"),
            nheads=hp.get("nheads"),
            dropout_attention_rate=hp.get("dropout_attention_rate"),
            num_layers_predictor=hp.get("num_layers_predictor"),
            dropout_predictor_rate=hp.get("dropout_predictor_rate"),
            dropout_fpdnn_rate=hp.get("dropout_fpdnn_rate"),
            gene_feat_size=self.gene_len,
            n_tasks=1,
            regression=(task == "regression"),
            args=args_stub,
            dropout_gene=hp.get("dropout_gene"),
            MLPu=[hp.get("MLPu1"), hp.get("MLPu2")],
        )

        self.model = SynProtXgatfp3_predictor(**cfg).to(self.device)

        # --------------- load weights (tolerant mode) -----------------------
        missing, _ = self.model.load_state_dict(clean_sd, strict=False)
        if missing:
            print("[warn] missing keys:", missing[:5], "…")
        self.model.eval()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    @torch.inference_mode()
    def predict(self, smiles_a: str, smiles_b: str, cell_line: str) -> float:
        """Return synergy/response score for the given triple."""
        cell_line = cell_line.upper()
        if cell_line not in self.gene_df.index:
            raise KeyError(f"Cell‑line '{cell_line}' not found in expression table.")

        # tensors -----------------------------------------------------------
        g1, g2 = _graph_gen(smiles_a), _graph_gen(smiles_b)
        
        # ────────── 1. pad node features so both drugs share the same length ──────────
        max_nodes = max(g1.num_nodes, g2.num_nodes)
        
        def _pad_graph(g, target):
            if g.num_nodes < target:
                pad = torch.zeros(target - g.num_nodes, g.x.size(1))
                g.x = torch.cat([g.x, pad], dim=0)
            return g
        
        g1 = _pad_graph(g1, max_nodes)
        g2 = _pad_graph(g2, max_nodes)

        # ────────── 2. a single batch vector now matches **both** x1 and x2 ──────────
        batch = torch.zeros(max_nodes, dtype=torch.long)
        
        g1.x, g2.x = _pad(g1.x, self.atom_dim), _pad(g2.x, self.atom_dim)
        drr = _smiles_to_ecfp6(smiles_a, self.fp_len).unsqueeze(0).to(self.device)
        drc = _smiles_to_ecfp6(smiles_b, self.fp_len).unsqueeze(0).to(self.device)
        
        # ───────── gene & protein expression tensors ─────────
        gene_arr = self.gene_df.loc[cell_line].values.astype(np.float32)   #  (n_genes,) 1‑D
        prot_arr = self.prot_df.loc[cell_line].values.astype(np.float32)   #  (6688,)    1‑D
        gene = _expr_row_as_tensor(self.gene_df,  cell_line, self.gene_len,  self.device)
        prot = _expr_row_as_tensor(self.prot_df,  cell_line, self.prot_len,  self.device)

        data = Data(
            x=[g1.x, g2.x],
            edge_index=[g1.edge_index, g2.edge_index],
            batch=batch,
            gene_exp=gene,  # (1, gene_len)
            prot_exp=prot,  # (1, 6688)
            drr=drr,
            drc=drc,
        ).to(self.device)

        y_hat: Tensor = self.model(data)
        return float(torch.sigmoid(y_hat)) if self.task == "classification" else float(y_hat)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _load_expression_tables(self) -> None:
        """Load and align gene / protein pickles to model feature sizes."""
        data_root = Path("data") / "export"
        gene_path = data_root / "data_preprocessing_gene.pkl"
        prot_path = data_root / "data_preprocessing_protein.pkl"
        if not gene_path.exists() or not prot_path.exists():
            raise FileNotFoundError(
                "Gene/protein expression pickles not found in ./data/export/."
            )

        # Gene -------------------------------------------------------------
        gene_df = pd.read_pickle(gene_path).reset_index(drop=True)
        gene_df = gene_df.set_index("cell_line_name").iloc[:, : self.gene_len]
        self.gene_df = gene_df

        # Protein ----------------------------------------------------------
        prot_df = pd.read_pickle(prot_path).reset_index(drop=True)
        cols = prot_df.columns
        prot_df.columns = ["id", "cell_line_name"] + [
            c.split(";")[1].split("_")[0] for c in cols[2:]
        ]
        prot_df = prot_df.fillna(0)

        # keep only the first <prot_len> expression columns after the two IDs
        prot_wanted = 2 + self.prot_len          # 'id' + 'cell_line_name' + features
        prot_df = prot_df.iloc[:, :prot_wanted]

        # safety‑net: if somebody gives a pickle with *fewer* than prot_len columns
        if prot_df.shape[1] - 2 < self.prot_len:
            missing = self.prot_len - (prot_df.shape[1] - 2)
            prot_df = pd.concat(
                [prot_df, pd.DataFrame(
                    np.zeros((len(prot_df), missing)),
                    columns=[f"pad{i}" for i in range(missing)]
                )],
                axis=1,
            )

        self.prot_df = prot_df.set_index("cell_line_name").iloc[:, 1:]  # drop 'id'

# ----------------------------------------------------------------------------
# 5.  Command‑line interface
# ----------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser("SynProtX inference (stand‑alone)")
    parser.add_argument("--smi1", required=True, help="SMILES string of first compound")
    parser.add_argument("--smi2", required=True, help="SMILES string of second compound")
    parser.add_argument("--cell_line", required=True, help="Cell‑line identifier (e.g. MCF7)")
    parser.add_argument("--dataset", default="ALMANAC-Breast", choices=SynProtXInference.DATASETS,
                        help="Which SynProtX dataset checkpoint to use")
    parser.add_argument("--task", default="regression", choices=["classification", "regression"],
                        help="Prediction head to use: classification (sigmoid) or regression (raw score)")
    parser.add_argument("--device", default="cpu", help="cpu or cuda device string, e.g. cuda:0")
    # ‑‑thr is only meaningful for classification but harmless otherwise
    parser.add_argument("--thr", type=float, default=0.5,
                        help="Threshold for classifying synergy vs antagonism (classification task only)")
    args = parser.parse_args()

    predictor = SynProtXInference(args.task, args.dataset, args.device)
    score = predictor.predict(args.smi1, args.smi2, args.cell_line)

    breaker = "-" * 64
    print(breaker)
    print(f"Dataset: {args.dataset}")
    print(f"Cell Line: {args.cell_line}")
    print(f"Task: {args.task}")
    print(f"SMILES1: {args.smi1}")
    print(f"SMILES2: {args.smi2}")
    print(f"Predicted Value: {score:.4f}")
    if args.task == "classification":
        interpretation = "Synergism" if score >= args.thr else "Antagonism"
        print(f"Interpretation: {interpretation} (Threshold = {args.thr})")
    print(breaker)

if __name__ == "__main__":
    _cli()
