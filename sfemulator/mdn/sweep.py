"""MDN architecture sweep: grid search over components × capacity.

Runs leave-one-galaxy-out cross-validation for each architecture config,
saves per-fold and mean OOS NLL and R² to a CSV.

Usage:
    python -m sfemulator.mdn.sweep --snap-dir snapshots --out results/mdn_sweep.csv
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
from sfemulator.data.loader import load_all_galaxies, prepare_log_data, GALAXIES

FEATURE_INDICES = [0]

COMPONENTS = [3, 5, 8, 12, 16]
CAPACITIES = [
    {"n_layers": 2, "n_hidden": 64,  "label": "2x64"},
    {"n_layers": 3, "n_hidden": 64,  "label": "3x64"},
    {"n_layers": 3, "n_hidden": 128, "label": "3x128"},
    {"n_layers": 5, "n_hidden": 128, "label": "5x128"},
    {"n_layers": 5, "n_hidden": 256, "label": "5x256"},
]


def run_sweep(snap_dir: Path, out_path: Path, n_epochs: int, device: str) -> None:
    """Run the full grid search and write results to CSV."""
    data = load_all_galaxies(snap_dir)
    galaxy_names = list(GALAXIES.keys())

    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    all_curves = {}
    total_configs = len(COMPONENTS) * len(CAPACITIES)
    config_idx = 0

    for cap in CAPACITIES:
        for n_comp in COMPONENTS:
            config_idx += 1
            label = f"{cap['label']}_K{n_comp}"
            print(f"\n{'='*60}")
            print(f"Config {config_idx}/{total_configs}: {label}")
            print(f"  layers={cap['n_layers']}, hidden={cap['n_hidden']}, components={n_comp}")
            print(f"{'='*60}")

            fold_nlls = []
            fold_r2s = []
            fold_curves = {}

            for held_out in galaxy_names:
                train_names = [n for n in galaxy_names if n != held_out]
                t0 = time.time()

                X_train, y_train, _ = prepare_log_data(data, train_names, feature_indices=FEATURE_INDICES)
                scaler = StandardScaler().fit(X_train)
                X_scaled = scaler.transform(X_train).astype(np.float32)
                y_tr = y_train.astype(np.float32)

                model, losses = train_mdn(
                    X_scaled, y_tr,
                    n_hidden=cap["n_hidden"],
                    n_layers=cap["n_layers"],
                    n_components=n_comp,
                    lr=1e-3, n_epochs=n_epochs, batch_size=2048, device=device,
                )

                # Evaluate on held-out galaxy
                ctx_test, sfr_sd_test = data[held_out]
                log_ctx_test = np.log10(ctx_test[:, FEATURE_INDICES])
                log_sfr_test = np.log10(sfr_sd_test)
                finite_test = np.all(np.isfinite(log_ctx_test), axis=1) & np.isfinite(log_sfr_test)

                X_test_scaled = scaler.transform(log_ctx_test[finite_test]).astype(np.float32)
                X_test_t = torch.tensor(X_test_scaled).to(device)
                y_test_true = log_sfr_test[finite_test]
                y_test_t = torch.tensor(y_test_true.astype(np.float32)).to(device)

                y_pred = predict_mean(model, X_test_t)
                nll = predict_nll(model, X_test_t, y_test_t)

                if np.any(np.isnan(y_pred)) or np.isnan(nll):
                    r2 = float("nan")
                    nll = float("nan")
                    print(f"  {held_out:<12} NLL=     NaN  R²=     NaN  (diverged)")
                else:
                    r2 = r2_score(y_test_true, y_pred)
                    print(f"  {held_out:<12} NLL={nll:8.4f}  R²={r2:8.4f}  ({time.time() - t0:.1f}s)")

                fold_nlls.append(nll)
                fold_r2s.append(r2)
                fold_curves[held_out] = losses

            mean_nll = np.nanmean(fold_nlls)
            mean_r2 = np.nanmean(fold_r2s)
            print(f"  {'MEAN':<12} NLL={mean_nll:8.4f}  R²={mean_r2:8.4f}")

            row = {
                "capacity": cap["label"],
                "n_layers": cap["n_layers"],
                "n_hidden": cap["n_hidden"],
                "n_components": n_comp,
            }
            for i, name in enumerate(galaxy_names):
                row[f"nll_{name}"] = fold_nlls[i]
                row[f"r2_{name}"] = fold_r2s[i]
            row["nll_mean"] = mean_nll
            row["r2_mean"] = mean_r2
            rows.append(row)
            all_curves[label] = fold_curves

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {out_path}")

    # Save training curves as JSON alongside the CSV
    curves_path = out_path.with_suffix(".curves.json")
    with open(curves_path, "w") as f:
        json.dump(all_curves, f)
    print(f"Training curves saved to {curves_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="MDN architecture sweep")
    parser.add_argument("--snap-dir", type=Path, default=Path("snapshots"))
    parser.add_argument("--out", type=Path, default=Path("results/mdn_sweep.csv"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        args.device = "mps" if torch.backends.mps.is_available() else "cpu"

    run_sweep(args.snap_dir, args.out, args.epochs, args.device)


if __name__ == "__main__":
    main()
