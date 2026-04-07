"""MDN feature sweep: gas density + one feature at a time.

Fixed architecture (2 layers, 64 hidden, 8 components).
Runs leave-one-galaxy-out cross-validation for each feature set,
saves per-fold in-sample and OOS NLL to CSV, plus training curves.

Usage:
    python -m sfemulator.mdn.sweep_features --snap-dir snapshots --out results/mdn_feature_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from sfemulator.mdn.model import train_mdn, predict_mean, predict_nll
from sfemulator.data.loader import load_all_galaxies, prepare_log_data, GALAXIES, CONTEXT_NAMES

N_LAYERS = 2
N_HIDDEN = 64
N_COMPONENTS = 8

FEATURE_SETS = [
    {"name": "gas",             "indices": [0]},
    {"name": "gas+star",        "indices": [0, 1]},
    {"name": "gas+rotcurve",    "indices": [0, 2]},
    {"name": "gas+kappa",       "indices": [0, 3]},
    {"name": "gas+dm",          "indices": [0, 4]},
    {"name": "gas+kappa+dm",    "indices": [0, 3, 4]},
]


def run_sweep(snap_dir: Path, out_path: Path, n_epochs: int, device: str) -> None:
    """Run the feature sweep and write results to CSV."""
    data = load_all_galaxies(snap_dir)
    galaxy_names = list(GALAXIES.keys())

    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    all_curves = {}

    for feat_idx, feat_set in enumerate(FEATURE_SETS):
        label = feat_set["name"]
        indices = feat_set["indices"]
        feat_desc = ", ".join(CONTEXT_NAMES[i] for i in indices)
        print(f"\n{'='*60}")
        print(f"Feature set {feat_idx+1}/{len(FEATURE_SETS)}: {label}")
        print(f"  Features: {feat_desc}")
        print(f"  Architecture: {N_LAYERS}x{N_HIDDEN}, K={N_COMPONENTS}")
        print(f"{'='*60}")

        fold_oos_nlls = []
        fold_oos_r2s = []
        fold_is_nlls = []
        fold_is_r2s = []
        fold_curves = {}

        for held_out in galaxy_names:
            train_names = [n for n in galaxy_names if n != held_out]
            t0 = time.time()

            X_train, y_train_log, _ = prepare_log_data(data, train_names, feature_indices=indices)
            scaler = StandardScaler().fit(X_train)
            X_scaled = scaler.transform(X_train).astype(np.float32)
            y_tr = y_train_log.astype(np.float32)

            model, losses = train_mdn(
                X_scaled, y_tr,
                n_hidden=N_HIDDEN, n_layers=N_LAYERS, n_components=N_COMPONENTS,
                lr=1e-3, n_epochs=n_epochs, batch_size=2048, device=device,
            )

            # In-sample evaluation
            X_train_t = torch.tensor(X_scaled).to(device)
            y_train_t = torch.tensor(y_tr).to(device)
            y_train_pred = predict_mean(model, X_train_t)
            is_nll = predict_nll(model, X_train_t, y_train_t)
            if np.any(np.isnan(y_train_pred)) or np.isnan(is_nll):
                is_r2 = float("nan")
                is_nll = float("nan")
            else:
                is_r2 = r2_score(y_train_log, y_train_pred)

            # OOS evaluation
            ctx_test, sfr_sd_test = data[held_out]
            log_ctx_test = np.log10(ctx_test[:, indices])
            log_sfr_test = np.log10(sfr_sd_test)
            finite_test = np.all(np.isfinite(log_ctx_test), axis=1) & np.isfinite(log_sfr_test)

            X_test_scaled = scaler.transform(log_ctx_test[finite_test]).astype(np.float32)
            X_test_t = torch.tensor(X_test_scaled).to(device)
            y_test_true = log_sfr_test[finite_test]
            y_test_t = torch.tensor(y_test_true.astype(np.float32)).to(device)

            y_pred = predict_mean(model, X_test_t)
            oos_nll = predict_nll(model, X_test_t, y_test_t)

            if np.any(np.isnan(y_pred)) or np.isnan(oos_nll):
                oos_r2 = float("nan")
                oos_nll = float("nan")
                print(f"  {held_out:<12} diverged")
            else:
                oos_r2 = r2_score(y_test_true, y_pred)
                elapsed = time.time() - t0
                print(f"  {held_out:<12} IS: NLL={is_nll:8.4f} R²={is_r2:8.4f}  OOS: NLL={oos_nll:8.4f} R²={oos_r2:8.4f}  ({elapsed:.1f}s)")

            fold_oos_nlls.append(oos_nll)
            fold_oos_r2s.append(oos_r2)
            fold_is_nlls.append(is_nll)
            fold_is_r2s.append(is_r2)
            fold_curves[held_out] = losses

        mean_oos_nll = np.nanmean(fold_oos_nlls)
        mean_oos_r2 = np.nanmean(fold_oos_r2s)
        mean_is_nll = np.nanmean(fold_is_nlls)
        mean_is_r2 = np.nanmean(fold_is_r2s)
        gap = mean_oos_nll - mean_is_nll
        print(f"  {'MEAN':<12} IS: NLL={mean_is_nll:8.4f} R²={mean_is_r2:8.4f}  OOS: NLL={mean_oos_nll:8.4f} R²={mean_oos_r2:8.4f}  gap={gap:+.4f}")

        row = {"features": label, "feature_indices": str(indices)}
        for i, name in enumerate(galaxy_names):
            row[f"is_nll_{name}"] = fold_is_nlls[i]
            row[f"is_r2_{name}"] = fold_is_r2s[i]
            row[f"oos_nll_{name}"] = fold_oos_nlls[i]
            row[f"oos_r2_{name}"] = fold_oos_r2s[i]
        row["is_nll_mean"] = mean_is_nll
        row["is_r2_mean"] = mean_is_r2
        row["oos_nll_mean"] = mean_oos_nll
        row["oos_r2_mean"] = mean_oos_r2
        row["nll_gap"] = gap
        rows.append(row)
        all_curves[label] = fold_curves

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {out_path}")

    curves_path = out_path.with_suffix(".curves.json")
    with open(curves_path, "w") as f:
        json.dump(all_curves, f)
    print(f"Training curves saved to {curves_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="MDN feature sweep")
    parser.add_argument("--snap-dir", type=Path, default=Path("snapshots"))
    parser.add_argument("--out", type=Path, default=Path("results/mdn_feature_sweep.csv"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        args.device = "mps" if torch.backends.mps.is_available() else "cpu"

    run_sweep(args.snap_dir, args.out, args.epochs, args.device)


if __name__ == "__main__":
    main()
