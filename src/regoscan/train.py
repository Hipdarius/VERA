"""
Training script for Regoscan.

Handles two paths:
1. Baseline (PLSR + RandomForest) - good for regression accuracy.
2. CNN (1D ResNet) - better at mineral classification.

Example:
  python -m regoscan.train --model cnn --data data/synth_v2.csv --out runs/cnn_v2
"""

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from regoscan.datasets import (
    NumpyBundle,
    RegoscanSpectraDataset,
    sample_level_split,
    split_bundle,
    to_bundle,
)
from regoscan.io_csv import read_measurements_csv
from regoscan.models.cnn import RegoscanCNN, assert_input_size, count_params
from regoscan.models.plsr import (
    BaselineBundle,
    build_baseline_features,
    fit_baseline,
    save_baseline,
)
from regoscan.schema import (
    CLASS_TO_INDEX,
    INDEX_TO_CLASS,
    MINERAL_CLASSES,
    N_CLASSES,
    N_FEATURES_TOTAL,
)


def set_global_seed(seed: int) -> None:
    """Keep things deterministic for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def quick_metrics(
    pred_cls: np.ndarray, pred_ilm: np.ndarray, true_cls: np.ndarray, true_ilm: np.ndarray
) -> dict[str, float]:
    """Simple acc / rmse / r2 calc for logging."""
    acc = float((pred_cls == true_cls).mean()) if pred_cls.size else float("nan")
    if pred_ilm.size:
        residual = pred_ilm - true_ilm
        rmse = float(np.sqrt(np.mean(residual**2)))
        var = float(np.var(true_ilm))
        r2 = float(1.0 - np.var(residual) / var) if var > 0 else float("nan")
    else:
        rmse = float("nan")
        r2 = float("nan")
    return {"top1_acc": acc, "ilm_rmse": rmse, "ilm_r2": r2}


def _run_cv_plsr(
    df: pd.DataFrame,
    n_folds: int,
    n_components: int,
    n_estimators: int,
    seed: int,
) -> list[dict[str, float]]:
    """Run stratified k-fold CV for the PLSR baseline. Returns per-fold metrics."""
    from sklearn.model_selection import StratifiedGroupKFold

    sample_ids = df["sample_id"].to_numpy()
    mineral_classes = df["mineral_class"].map(CLASS_TO_INDEX).to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results: list[dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        sgkf.split(df, y=mineral_classes, groups=sample_ids)
    ):
        train_bundle = to_bundle(df.iloc[train_idx])
        val_bundle = to_bundle(df.iloc[val_idx])

        bb = fit_baseline(
            train_bundle,
            n_components=n_components,
            n_estimators=n_estimators,
            seed=seed,
        )
        X_val = build_baseline_features(val_bundle)
        pred_cls, pred_ilm = bb.predict(X_val)
        m = quick_metrics(pred_cls, pred_ilm, val_bundle.class_idx, val_bundle.ilmenite)
        m["fold"] = fold_idx
        fold_results.append(m)
        print(
            f"  fold {fold_idx + 1}/{n_folds}  "
            f"acc={m['top1_acc']:.3f}  rmse={m['ilm_rmse']:.3f}  R2={m['ilm_r2']:.3f}"
        )

    return fold_results


def _run_cv_cnn(
    df: pd.DataFrame,
    n_folds: int,
    args: argparse.Namespace,
) -> list[dict[str, float]]:
    """Run stratified k-fold CV for the CNN. Returns per-fold metrics."""
    from sklearn.model_selection import StratifiedGroupKFold

    sample_ids = df["sample_id"].to_numpy()
    mineral_classes = df["mineral_class"].map(CLASS_TO_INDEX).to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
    fold_results: list[dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        sgkf.split(df, y=mineral_classes, groups=sample_ids)
    ):
        set_global_seed(args.seed)
        train_bundle = to_bundle(df.iloc[train_idx])
        val_bundle = to_bundle(df.iloc[val_idx])

        train_ds = RegoscanSpectraDataset(train_bundle, augment=True, seed=args.seed)
        val_ds = RegoscanSpectraDataset(val_bundle, augment=False)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        model = RegoscanCNN(dropout=args.dropout, seed=args.seed)
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=max(1, args.epochs), eta_min=args.lr * 0.02
        )
        cls_loss_fn = torch.nn.CrossEntropyLoss()
        reg_loss_fn = torch.nn.SmoothL1Loss()

        best_val_acc = -1.0
        best_state = None

        for epoch in range(1, args.epochs + 1):
            model.train()
            for x, y_cls, y_ilm in train_loader:
                optim.zero_grad()
                logits, ilm = model(x)
                loss = cls_loss_fn(logits, y_cls) + args.reg_weight * reg_loss_fn(ilm, y_ilm)
                loss.backward()
                optim.step()
            scheduler.step()
            val_m = _eval_loader(model, val_loader)
            if val_m["top1_acc"] > best_val_acc + 1e-6:
                best_val_acc = val_m["top1_acc"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Evaluate best checkpoint
        if best_state is not None:
            model.load_state_dict(best_state)
        m = _eval_loader(model, val_loader)
        m["fold"] = fold_idx
        fold_results.append(m)
        print(
            f"  fold {fold_idx + 1}/{n_folds}  "
            f"acc={m['top1_acc']:.3f}  rmse={m['ilm_rmse']:.3f}  R2={m['ilm_r2']:.3f}"
        )

    return fold_results


def _summarise_cv(fold_results: list[dict[str, float]], out: Path) -> dict:
    """Compute mean +/- std of CV metrics, write to cv_results.json, return summary."""
    accs = np.array([f["top1_acc"] for f in fold_results])
    rmses = np.array([f["ilm_rmse"] for f in fold_results])
    r2s = np.array([f["ilm_r2"] for f in fold_results])

    summary = {
        "n_folds": len(fold_results),
        "per_fold": fold_results,
        "mean_accuracy": float(accs.mean()),
        "std_accuracy": float(accs.std()),
        "mean_rmse": float(rmses.mean()),
        "std_rmse": float(rmses.std()),
        "mean_r2": float(r2s.mean()),
        "std_r2": float(r2s.std()),
    }
    (out / "cv_results.json").write_text(json.dumps(summary, indent=2))

    print(f"\nCross-validation summary ({len(fold_results)} folds):")
    print(f"  accuracy: {accs.mean():.3f} +/- {accs.std():.3f}")
    print(f"  RMSE:     {rmses.mean():.3f} +/- {rmses.std():.3f}")
    print(f"  R2:       {r2s.mean():.3f} +/- {r2s.std():.3f}")
    return summary


def run_plsr(args: argparse.Namespace) -> int:
    """The statistical baseline path."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)
    df = read_measurements_csv(args.data)
    print(f"Loaded {len(df)} measurements")

    # Split by sample_id so we don't leak info between train/test
    split = sample_level_split(
        df, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    )
    bundles = split_bundle(df, split)
    print(f"Split: train={len(bundles['train'].class_idx)}, val={len(bundles['val'].class_idx)}, test={len(bundles['test'].class_idx)}")

    # Hyperparams for PLSR: 8 components seems to be the sweet spot for VIS/NIR.
    # More than 12 starts to overfit the synthetic noise.
    n_components = args.pls_components
    n_estimators = args.rf_estimators
    bundle = fit_baseline(
        bundles["train"],
        n_components=n_components,
        n_estimators=n_estimators,
        seed=args.seed,
    )
    save_baseline(bundle, out / "model.pkl")

    metrics: dict[str, dict[str, float]] = {}
    for name in ("train", "val", "test"):
        b: NumpyBundle = bundles[name]
        if len(b.class_idx) == 0:
            metrics[name] = {"top1_acc": float("nan"), "ilm_rmse": float("nan"), "ilm_r2": float("nan")}
            continue
        X = build_baseline_features(b)
        pred_cls, pred_ilm = bundle.predict(X)
        metrics[name] = quick_metrics(pred_cls, pred_ilm, b.class_idx, b.ilmenite)

    manifest = {
        "model": "plsr",
        "data": str(Path(args.data).resolve()),
        "seed": args.seed,
        "metrics": metrics,
        "n_components": n_components,
        "n_estimators": n_estimators,
    }
    (out / "run.json").write_text(json.dumps(manifest, indent=2))
    (out / "split.json").write_text(
        json.dumps({
            "train_samples": list(split.train_samples),
            "val_samples": list(split.val_samples),
            "test_samples": list(split.test_samples),
        }, indent=2)
    )

    # Optional k-fold cross-validation
    if args.cv_folds > 0:
        print(f"\nRunning {args.cv_folds}-fold stratified CV (PLSR)...")
        fold_results = _run_cv_plsr(
            df,
            n_folds=args.cv_folds,
            n_components=n_components,
            n_estimators=n_estimators,
            seed=args.seed,
        )
        _summarise_cv(fold_results, out)

    print("\nPLSR Baseline Metrics:")
    for k, v in metrics.items():
        print(f"  {k:5s}  acc={v['top1_acc']:.3f}  rmse={v['ilm_rmse']:.3f}  R2={v['ilm_r2']:.3f}")
    return 0


def run_cnn(args: argparse.Namespace) -> int:
    """1D ResNet path. Much slower but hits higher classification accuracy."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)
    assert_input_size()

    df = read_measurements_csv(args.data)
    print(f"Loaded {len(df)} measurements")

    split = sample_level_split(df, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)
    bundles = split_bundle(df, split)

    # Use heavy data augmentation only for training.
    train_ds = RegoscanSpectraDataset(bundles["train"], augment=True, seed=args.seed)
    val_ds = RegoscanSpectraDataset(bundles["val"], augment=False)
    test_ds = RegoscanSpectraDataset(bundles["test"], augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = RegoscanCNN(dropout=args.dropout, seed=args.seed)
    print(f"CNN parameters: {count_params(model):,}")
    
    # AdamW + Cosine Annealing is a solid combo for this type of spectral data.
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(1, args.epochs), eta_min=args.lr * 0.02
    )

    cls_loss_fn = torch.nn.CrossEntropyLoss()
    reg_loss_fn = torch.nn.SmoothL1Loss()

    history: list[dict[str, float]] = []
    best_val_acc = -1.0
    epochs_without_improvement = 0
    patience = args.early_stopping_patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for x, y_cls, y_ilm in train_loader:
            optim.zero_grad()
            logits, ilm = model(x)
            # Weighted loss: classification is primary, regression is secondary but important.
            loss = cls_loss_fn(logits, y_cls) + args.reg_weight * reg_loss_fn(ilm, y_ilm)
            loss.backward()
            optim.step()
            running_loss += float(loss.item())
            n_batches += 1
        
        scheduler.step()
        train_loss = running_loss / max(n_batches, 1)
        val_metrics = _eval_loader(model, val_loader)
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": float(optim.param_groups[0]["lr"]),
            **val_metrics,
        })

        # Early stopping logic
        if val_metrics["top1_acc"] > best_val_acc + 1e-6:
            best_val_acc = val_metrics["top1_acc"]
            torch.save(model.state_dict(), out / "model.pt")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:3d}/{args.epochs} - loss: {train_loss:.3f}, val_acc: {val_metrics['top1_acc']:.3f}, val_rmse: {val_metrics['ilm_rmse']:.3f}")

        if patience > 0 and epochs_without_improvement >= patience:
            print(f"[Early Stop] No improvement for {patience} epochs. Best val_acc: {best_val_acc:.3f}")
            break

    # Reload best and get final scores
    model.load_state_dict(torch.load(out / "model.pt"))
    final_metrics = {
        "train": _eval_loader(model, DataLoader(train_ds, batch_size=args.batch_size)),
        "val": _eval_loader(model, val_loader),
        "test": _eval_loader(model, test_loader),
    }

    # Save metadata for the API/quantization steps
    manifest = {
        "model": "cnn",
        "epochs_completed": epoch,
        "metrics": final_metrics,
        "history": history,
        "input_shape": [1, N_FEATURES_TOTAL],
    }
    (out / "run.json").write_text(json.dumps(manifest, indent=2))
    (out / "meta.json").write_text(json.dumps({
        "n_classes": N_CLASSES,
        "class_names": list(MINERAL_CLASSES),
        "input_shape": [1, N_FEATURES_TOTAL],
    }, indent=2))
    (out / "split.json").write_text(
        json.dumps({
            "train_samples": list(split.train_samples),
            "val_samples": list(split.val_samples),
            "test_samples": list(split.test_samples),
        }, indent=2)
    )
    
    # Optional k-fold cross-validation
    if args.cv_folds > 0:
        print(f"\nRunning {args.cv_folds}-fold stratified CV (CNN)...")
        fold_results = _run_cv_cnn(df, n_folds=args.cv_folds, args=args)
        _summarise_cv(fold_results, out)

    print("\nCNN Final Metrics:")
    for k, v in final_metrics.items():
        print(f"  {k:5s}  acc={v['top1_acc']:.3f}  rmse={v['ilm_rmse']:.3f}  R2={v['ilm_r2']:.3f}")

    try:
        _plot_training_history(history, out / "training_history.png")
    except Exception as e:
        print(f"Plotting failed: {e}")
    return 0


def _plot_training_history(history: list[dict[str, float]], out_path: Path) -> None:
    """Quick plot of the training curve."""
    import matplotlib.pyplot as plt
    if not history: return
    
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_acc = [h["top1_acc"] for h in history]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, train_loss, color="blue", label="train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="blue")
    
    ax2 = ax1.twinx()
    ax2.plot(epochs, val_acc, color="orange", label="val acc")
    ax2.set_ylabel("Accuracy", color="orange")
    
    plt.title("Training History")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _eval_loader(model: RegoscanCNN, loader: DataLoader) -> dict[str, float]:
    model.eval()
    pred_cls_all, pred_ilm_all = [], []
    true_cls_all, true_ilm_all = [], []
    
    with torch.no_grad():
        for x, y_cls, y_ilm in loader:
            logits, ilm = model(x)
            pred_cls_all.extend(logits.argmax(dim=1).tolist())
            pred_ilm_all.extend(ilm.tolist())
            true_cls_all.extend(y_cls.tolist())
            true_ilm_all.extend(y_ilm.tolist())
            
    return quick_metrics(
        np.asarray(pred_cls_all), np.asarray(pred_ilm_all),
        np.asarray(true_cls_all), np.asarray(true_ilm_all),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a Regoscan model")
    p.add_argument("--model", choices=["plsr", "cnn"], required=True)
    p.add_argument("--data", required=True, help="path to a Regoscan CSV")
    p.add_argument("--out", required=True, help="output run directory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--early-stopping-patience", type=int, default=15)
    # Configurable hyperparameters (defaults match original hardcoded values)
    p.add_argument("--reg-weight", type=float, default=0.5,
                   help="regression loss multiplier (default: 0.5)")
    p.add_argument("--rf-estimators", type=int, default=200,
                   help="RandomForest n_estimators (default: 200)")
    p.add_argument("--pls-components", type=int, default=8,
                   help="PLS n_components (default: 8)")
    p.add_argument("--dropout", type=float, default=0.25,
                   help="CNN dropout rate (default: 0.25)")
    p.add_argument("--cv-folds", type=int, default=0,
                   help="number of stratified k-fold CV folds (0=disabled, use hold-out)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.model == "plsr":
        return run_plsr(args)
    return run_cnn(args)


if __name__ == "__main__":
    raise SystemExit(main())
