"""Training entry point for Regoscan models.

Two model paths share one CLI:

  python -m regoscan.train --model plsr --data <csv> --out <run_dir>
  python -m regoscan.train --model cnn  --data <csv> --epochs 20 --out <run_dir>

Both write the following into ``<run_dir>``:

  - ``run.json``           manifest with model type, hyperparams, metrics
  - ``split.json``         the exact train/val/test sample_id partition
  - ``model.pkl`` or ``model.pt``  the fitted artefact
  - For CNN: ``meta.json`` with input shape and class index ↔ name mapping
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
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
    INDEX_TO_CLASS,
    MINERAL_CLASSES,
    N_CLASSES,
    N_FEATURES_TOTAL,
)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Metrics (light — full eval lives in evaluate.py)
# ---------------------------------------------------------------------------


def quick_metrics(
    pred_cls: np.ndarray, pred_ilm: np.ndarray, true_cls: np.ndarray, true_ilm: np.ndarray
) -> dict[str, float]:
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


# ---------------------------------------------------------------------------
# PLSR / RandomForest path
# ---------------------------------------------------------------------------


def run_plsr(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)
    df = read_measurements_csv(args.data)
    print(f"[ok] loaded {len(df)} measurements from {args.data}")

    split = sample_level_split(
        df, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    )
    bundles = split_bundle(df, split)
    print(
        f"[ok] split: train={len(bundles['train'].class_idx)} "
        f"val={len(bundles['val'].class_idx)} "
        f"test={len(bundles['test'].class_idx)}"
    )

    bundle = fit_baseline(bundles["train"], n_components=8, n_estimators=200, seed=args.seed)
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
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "n_components": 8,
        "n_estimators": 200,
        "metrics": metrics,
        "feature_dim": int(build_baseline_features(bundles["train"]).shape[1]),
    }
    (out / "run.json").write_text(json.dumps(manifest, indent=2))
    (out / "split.json").write_text(
        json.dumps(
            {
                "train_samples": list(split.train_samples),
                "val_samples": list(split.val_samples),
                "test_samples": list(split.test_samples),
            },
            indent=2,
        )
    )

    print("[ok] PLSR baseline metrics:")
    for k, v in metrics.items():
        print(f"  {k:5s}  acc={v['top1_acc']:.3f}  rmse={v['ilm_rmse']:.3f}  R2={v['ilm_r2']:.3f}")
    print(f"[ok] wrote {out / 'model.pkl'}")
    return 0


# ---------------------------------------------------------------------------
# CNN path
# ---------------------------------------------------------------------------


def run_cnn(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)
    assert_input_size()

    df = read_measurements_csv(args.data)
    print(f"[ok] loaded {len(df)} measurements from {args.data}")

    split = sample_level_split(
        df, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    )
    bundles = split_bundle(df, split)
    print(
        f"[ok] split: train={len(bundles['train'].class_idx)} "
        f"val={len(bundles['val'].class_idx)} "
        f"test={len(bundles['test'].class_idx)}"
    )

    train_ds = RegoscanSpectraDataset(bundles["train"], augment=True, seed=args.seed)
    val_ds = RegoscanSpectraDataset(bundles["val"], augment=False)
    test_ds = RegoscanSpectraDataset(bundles["test"], augment=False)

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, generator=g, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = RegoscanCNN(seed=args.seed)
    print(f"[ok] CNN params: {count_params(model):,}")
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # LR scheduler: cosine annealing over the full epoch budget.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(1, args.epochs), eta_min=args.lr * 0.02
    )

    cls_loss_fn = torch.nn.CrossEntropyLoss()
    reg_loss_fn = torch.nn.SmoothL1Loss()

    history: list[dict[str, float]] = []
    best_val_acc = -1.0
    epochs_without_improvement = 0
    patience = args.early_stopping_patience
    stopped_at_epoch = args.epochs

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for x, y_cls, y_ilm in train_loader:
            optim.zero_grad()
            logits, ilm = model(x)
            loss = cls_loss_fn(logits, y_cls) + 0.5 * reg_loss_fn(ilm, y_ilm)
            loss.backward()
            optim.step()
            running += float(loss.item())
            n_batches += 1
        scheduler.step()
        train_loss = running / max(n_batches, 1)

        val_metrics = _eval_loader(model, val_loader)
        current_lr = float(optim.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "lr": current_lr,
                **val_metrics,
            }
        )
        if val_metrics["top1_acc"] > best_val_acc + 1e-6:
            best_val_acc = val_metrics["top1_acc"]
            torch.save(model.state_dict(), out / "model.pt")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
            print(
                f"  epoch {epoch:3d}/{args.epochs}  "
                f"loss={train_loss:.3f}  "
                f"val_acc={val_metrics['top1_acc']:.3f}  "
                f"val_rmse={val_metrics['ilm_rmse']:.3f}  "
                f"lr={current_lr:.1e}"
            )

        if patience > 0 and epochs_without_improvement >= patience:
            stopped_at_epoch = epoch
            print(
                f"[early stop] no val improvement for {patience} epochs; "
                f"best val_acc={best_val_acc:.3f} at epoch {epoch - patience}"
            )
            break

    # final eval on best checkpoint
    model.load_state_dict(torch.load(out / "model.pt"))
    final_metrics = {
        "train": _eval_loader(model, DataLoader(train_ds, batch_size=args.batch_size)),
        "val": _eval_loader(model, val_loader),
        "test": _eval_loader(model, test_loader),
    }

    manifest = {
        "model": "cnn",
        "data": str(Path(args.data).resolve()),
        "seed": args.seed,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "epochs": args.epochs,
        "epochs_completed": stopped_at_epoch,
        "early_stopping_patience": patience,
        "lr": args.lr,
        "lr_scheduler": "CosineAnnealingLR",
        "batch_size": args.batch_size,
        "n_params": count_params(model),
        "metrics": final_metrics,
        "history": history,
        "input_shape": [1, N_FEATURES_TOTAL],
        "best_val_acc": best_val_acc,
    }
    (out / "run.json").write_text(json.dumps(manifest, indent=2))
    (out / "meta.json").write_text(
        json.dumps(
            {
                "n_classes": N_CLASSES,
                "class_names": list(MINERAL_CLASSES),
                "input_shape": [1, N_FEATURES_TOTAL],
            },
            indent=2,
        )
    )
    (out / "split.json").write_text(
        json.dumps(
            {
                "train_samples": list(split.train_samples),
                "val_samples": list(split.val_samples),
                "test_samples": list(split.test_samples),
            },
            indent=2,
        )
    )

    print("[ok] CNN final metrics:")
    for k, v in final_metrics.items():
        print(f"  {k:5s}  acc={v['top1_acc']:.3f}  rmse={v['ilm_rmse']:.3f}  R2={v['ilm_r2']:.3f}")
    print(f"[ok] wrote {out / 'model.pt'}")

    # training-history plot
    try:
        _plot_training_history(history, out / "training_history.png")
        print(f"[ok] wrote plot: {out / 'training_history.png'}")
    except Exception as e:  # plotting is a nice-to-have, not load-bearing
        print(f"[warn] failed to render training_history.png: {e}")
    return 0


def _plot_training_history(history: list[dict[str, float]], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        return
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_acc = [h["top1_acc"] for h in history]
    val_rmse = [h["ilm_rmse"] for h in history]
    lrs = [h.get("lr", float("nan")) for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=140)
    ax1.plot(epochs, train_loss, color="#00d1ff", label="train loss")
    ax1b = ax1.twinx()
    ax1b.plot(epochs, val_acc, color="#f5a623", label="val acc")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color="#00d1ff")
    ax1b.set_ylabel("val top-1 acc", color="#f5a623")
    ax1.set_title("Training loss / val accuracy")
    ax1.grid(alpha=0.25)

    ax2.plot(epochs, val_rmse, color="#00d1ff", label="val ilm RMSE")
    ax2b = ax2.twinx()
    ax2b.plot(epochs, lrs, color="#888", label="lr", linestyle="--")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("val ilm RMSE", color="#00d1ff")
    ax2b.set_ylabel("learning rate", color="#888")
    ax2.set_title("Val RMSE + LR schedule")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _eval_loader(model: RegoscanCNN, loader: DataLoader) -> dict[str, float]:
    model.eval()
    pred_cls_all: list[int] = []
    pred_ilm_all: list[float] = []
    true_cls_all: list[int] = []
    true_ilm_all: list[float] = []
    with torch.no_grad():
        for x, y_cls, y_ilm in loader:
            logits, ilm = model(x)
            pred_cls_all.extend(int(c) for c in logits.argmax(dim=1).tolist())
            pred_ilm_all.extend(float(v) for v in ilm.tolist())
            true_cls_all.extend(int(c) for c in y_cls.tolist())
            true_ilm_all.extend(float(v) for v in y_ilm.tolist())
    return quick_metrics(
        np.asarray(pred_cls_all),
        np.asarray(pred_ilm_all),
        np.asarray(true_cls_all),
        np.asarray(true_ilm_all),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a Regoscan model")
    p.add_argument("--model", choices=["plsr", "cnn"], required=True)
    p.add_argument("--data", required=True, help="path to a Regoscan CSV")
    p.add_argument("--out", required=True, help="output run directory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    # CNN-only knobs
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=15,
        help="stop training if val_acc doesn't improve for this many epochs (0 to disable)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.model == "plsr":
        return run_plsr(args)
    return run_cnn(args)


if __name__ == "__main__":
    raise SystemExit(main())
