"""
04_training/optuna_search.py
════════════════════════════════════════════════════════════════
Optuna [REMOVED_ZH:5]
[REMOVED_ZH:4]: hidden_dim, n_rgcn_layers, lstm_hidden, lr, dropout, lambdas

Run:
  python optuna_search.py --n_trials 50 --epochs 30
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_graph_construction"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03_model"))

import torch
import numpy as np
import pickle

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA = True
except ImportError:
    _OPTUNA = False
    print("[Optuna] optuna [REMOVED_ZH:3]: pip install optuna")

from dataset    import UTCIGraphDataset
from urbangraph import UrbanGraph
from train      import Trainer, build_env_time_seq


def objective(trial, h5_path, scenario_pkl, epw_pkl, n_epochs, device):
    """[REMOVED_ZH:2] trial [REMOVED_ZH:5] val_loss。"""
    # [REMOVED_ZH:4]
    hidden_dim    = trial.suggest_categorical("hidden_dim",    [64, 128, 256, 384])
    n_rgcn_layers = trial.suggest_int("n_rgcn_layers",         2, 4)
    lstm_hidden   = trial.suggest_categorical("lstm_hidden",   [128, 256, 512])
    lr            = trial.suggest_float("lr",                  1e-4, 1e-2, log=True)
    dropout       = trial.suggest_float("dropout",             0.05, 0.3)
    lambda1       = trial.suggest_float("lambda1",             0.01, 0.5)
    lambda2       = trial.suggest_float("lambda2",             0.01, 0.2)

    cfg = {
        "model": {
            "hidden_dim":    hidden_dim,
            "n_rgcn_layers": n_rgcn_layers,
            "lstm_hidden":   lstm_hidden,
            "dropout":       dropout,
            "lambdas":       {"lambda1": lambda1, "lambda2": lambda2, "lambda3": 0.05},
        },
        "training": {
            "lr": lr,
            "weight_decay": 1e-5,
            "max_epochs":   n_epochs,
            "early_stopping_patience": n_epochs,   # [REMOVED_ZH:8]
        }
    }

    with open(epw_pkl, "rb") as f:
        epw = pickle.load(f)

    train_ds = UTCIGraphDataset(h5_path, scenario_pkl, epw_pkl, split="train")
    val_ds   = UTCIGraphDataset(h5_path, scenario_pkl, epw_pkl, split="val")
    cfg["model"]["out_timesteps"] = len(train_ds.sim_hours)

    model   = UrbanGraph(**{k: cfg["model"][k] for k in
                             ["hidden_dim","n_rgcn_layers","lstm_hidden",
                              "out_timesteps","dropout"]},
                          lambdas=cfg["model"]["lambdas"])

    trainer = Trainer(model, train_ds, val_ds, epw, cfg, device,
                       out_dir=f"optuna_ckpts/trial_{trial.number}")
    history = trainer.fit()

    return min(history["val_loss"]) if history["val_loss"] else float("inf")


def main(n_trials:    int = 50,
          n_epochs:   int = 30,
          h5_path:    str = "../../01_data_generation/outputs/raw_simulations/ground_truth.h5",
          scenario_pkl:str = "../../01_data_generation/outputs/raw_simulations/scenarios.pkl",
          epw_pkl:    str = "../../01_data_generation/outputs/raw_simulations/epw_data.pkl",
          out_json:   str = "optuna_best_params.json"):

    if not _OPTUNA:
        print("[REMOVED_ZH:3] optuna: pip install optuna")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Optuna] n_trials={n_trials}  n_epochs={n_epochs}  device={device}")

    study = optuna.create_study(
        direction="minimize",
        study_name="urbangraph_hpo",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )

    study.optimize(
        lambda t: objective(t, h5_path, scenario_pkl, epw_pkl, n_epochs, device),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_params
    print(f"\n[Optuna] [REMOVED_ZH:2] val_loss = {study.best_value:.4f}")
    print(f"  [REMOVED_ZH:5]: {json.dumps(best, indent=2)}")

    with open(out_json, "w") as f:
        json.dump({"best_params": best, "best_val_loss": study.best_value}, f, indent=2)
    print(f"  [REMOVED_ZH:3]: {out_json}")
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trials", type=int,  default=50)
    ap.add_argument("--epochs",   type=int,  default=30)
    ap.add_argument("--h5",       default="../../01_data_generation/outputs/raw_simulations/ground_truth.h5")
    ap.add_argument("--scenarios",default="../../01_data_generation/outputs/raw_simulations/scenarios.pkl")
    ap.add_argument("--epw",      default="../../01_data_generation/outputs/raw_simulations/epw_data.pkl")
    ap.add_argument("--out",      default="optuna_best_params.json")
    args = ap.parse_args()
    main(args.n_trials, args.epochs, args.h5, args.scenarios, args.epw, args.out)