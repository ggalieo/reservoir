"""RandomForest 基线：同时支持回归（孔隙度）与分类（岩性）。

划分一律按井分组：同一口井的窗口不会同时出现在训练与测试。
    井数少（<10）  -> 留一井（LOO）
    井数多（>=10） -> 按井分组的 K 折（GroupKFold）

用法：
    python -m src.baseline_rf --dataset s1_demo              # 回归
    python -m src.baseline_rf --dataset force2020 --folds 5  # 分类
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

STATS = ("mean", "std", "min", "max", "p10", "p90")


def window_features(win: np.ndarray, channels: list[str]):
    n, c, _ = win.shape
    cols, names = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for ci, ch in enumerate(channels):
            b = win[:, ci, :].astype(np.float64)
            cols += [
                np.nanmean(b, axis=1),
                np.nanstd(b, axis=1),
                np.nanmin(b, axis=1),
                np.nanmax(b, axis=1),
                np.nanpercentile(b, 10, axis=1),
                np.nanpercentile(b, 90, axis=1),
            ]
            names += [f"{ch}_{s}" for s in STATS]
    return np.column_stack(cols), names


def impute_train_only(Xtr, Xte):
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return (
        np.where(np.isfinite(Xtr), Xtr, med),
        np.where(np.isfinite(Xte), Xte, med),
    )


def make_folds(wells: np.ndarray, n_folds: int):
    uniq = np.array(sorted(set(wells.tolist())))
    if uniq.size <= n_folds:
        return [(w, np.where(wells != w)[0], np.where(wells == w)[0]) for w in uniq]
    # 按井分组分块，保证同一口井整体落在一折
    folds = [[] for _ in range(n_folds)]
    for i, w in enumerate(uniq):
        folds[i % n_folds].append(w)
    out = []
    for k, ws in enumerate(folds):
        te = np.isin(wells, ws)
        out.append((f"fold{k}（{len(ws)} 井）", np.where(~te)[0], np.where(te)[0]))
    return out


def run_regression(y, X, wells, feat_names, args):
    from sklearn.metrics import r2_score as _r2

    folds, pt, pp = [], [], []
    for seed in range(args.seeds):
        for name, tr, te in make_folds(wells, args.folds):
            Xtr, Xte = impute_train_only(X[tr], X[te])
            m = RandomForestRegressor(
                n_estimators=args.n_estimators,
                min_samples_leaf=3,
                random_state=seed,
                n_jobs=-1,
            )
            m.fit(Xtr, y[tr])
            pred = m.predict(Xte)
            folds.append(
                {
                    "fold": name,
                    "seed": seed,
                    "n": int(te.size),
                    "rmse": float(np.sqrt(mean_squared_error(y[te], pred))),
                    "r2": float(_r2(y[te], pred)),
                }
            )
            pt.append(y[te])
            pp.append(pred)
    pt, pp = np.concatenate(pt), np.concatenate(pp)
    overall = {
        "rmse": float(np.sqrt(mean_squared_error(pt, pp))),
        "mae": float(mean_absolute_error(pt, pp)),
        "r2": float(r2_score(pt, pp)),
        "n": int(pt.size),
    }
    print(f"\n总体: RMSE {overall['rmse']:.4f}  R2 {overall['r2']:.3f}  n={overall['n']}")
    return overall, folds, pt, pp, feat_names


def run_classification(y, X, wells, feat_names, args, class_ids):
    folds, pt, pp = [], [], []
    t0 = time.time()
    for seed in range(args.seeds):
        for name, tr, te in make_folds(wells, args.folds):
            Xtr, Xte = impute_train_only(X[tr], X[te])
            m = RandomForestClassifier(
                n_estimators=args.n_estimators,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            )
            m.fit(Xtr, y[tr])
            pred = m.predict(Xte)
            folds.append(
                {
                    "fold": name,
                    "seed": seed,
                    "n": int(te.size),
                    "acc": float(accuracy_score(y[te], pred)),
                    "bacc": float(balanced_accuracy_score(y[te], pred)),
                    "f1_macro": float(f1_score(y[te], pred, average="macro", zero_division=0)),
                }
            )
            pt.append(y[te])
            pp.append(pred)
            print(
                f"  seed{seed} {name}: acc {folds[-1]['acc']:.3f} "
                f"bacc {folds[-1]['bacc']:.3f} f1m {folds[-1]['f1_macro']:.3f}"
            )
    pt, pp = np.concatenate(pt).astype(int), np.concatenate(pp).astype(int)

    labels = class_ids.tolist()
    overall = {
        "acc": float(accuracy_score(pt, pp)),
        "bacc": float(balanced_accuracy_score(pt, pp)),
        "f1_macro": float(f1_score(pt, pp, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(pt, pp, average="weighted", zero_division=0)),
        "n": int(pt.size),
        "elapsed_s": round(time.time() - t0, 1),
    }
    rec = recall_score(pt, pp, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(pt, pp, labels=labels)
    print(
        f"\n总体: acc {overall['acc']:.3f}  bacc {overall['bacc']:.3f}  "
        f"F1-macro {overall['f1_macro']:.3f}  n={overall['n']}"
    )
    print(f"\n{'类别':>8}{'测试数':>8}{'召回':>8}")
    for c, r in zip(labels, rec):
        print(f"{c:>8}{int((pt == c).sum()):>8}{r:>8.3f}")
    return overall, folds, pt, pp, cm, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="s1_demo")
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    path = os.path.join("data", "processed", f"{args.dataset}_labeled_windows.npz")
    if not os.path.exists(path):
        sys.exit(f"找不到 {path}，先运行 build_dataset.py --dataset {args.dataset}")

    d = np.load(path, allow_pickle=True)
    task = str(d["task"][0]) if "task" in d.files else "regression"
    win, y, wells = d["win"], d["y"], d["well"].astype(str)
    channels = d["channels"].tolist()
    class_ids = d["class_ids"] if "class_ids" in d.files else np.array([])

    mask = np.isfinite(y) & np.isfinite(win).any(axis=(1, 2))
    win, y, wells = win[mask], y[mask], wells[mask]
    print(f"== {args.dataset}  任务={task} ==")
    print(f"   参与建模 {mask.sum():,} 窗口，{len(set(wells.tolist()))} 口井")

    X, names = window_features(win, channels)
    print(f"   特征 {X.shape[1]} 维")
    y = y.astype(int) if task == "classification" else y

    if task == "classification":
        overall, folds, pt, pp, cm, rec = run_classification(y, X, wells, names, args, class_ids)
        out = {
            "dataset": args.dataset,
            "task": task,
            "overall": overall,
            "folds": folds,
            "class_ids": class_ids.tolist(),
            "recall_per_class": {int(c): float(r) for c, r in zip(class_ids.tolist(), rec)},
            "confusion_matrix": cm.tolist(),
        }
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        im = ax.imshow(cm, cmap="Blues", norm=matplotlib.colors.LogNorm(vmin=1))
        ax.set_xticks(range(len(class_ids)))
        ax.set_yticks(range(len(class_ids)))
        ax.set_xticklabels([str(c) for c in class_ids], rotation=45, fontsize=7)
        ax.set_yticklabels([str(c) for c in class_ids], fontsize=7)
        ax.set_xlabel("预测")
        ax.set_ylabel("实测")
        ax.set_title(f"岩性分类混淆矩阵  acc={overall['acc']:.3f}")
        fig.colorbar(im, ax=ax, label="窗口数")
        fig.tight_layout()
        png = os.path.join("reports", f"{args.dataset}_baseline_cm.png")
        fig.savefig(png, dpi=140)
        plt.close(fig)
    else:
        overall, folds, pt, pp, names = run_regression(y, X, wells, names, args)
        out = {"dataset": args.dataset, "task": task, "overall": overall, "folds": folds}
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(pt, pp, s=12, alpha=0.6)
        lim = [float(min(pt.min(), pp.min())), float(max(pt.max(), pp.max()))]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel("实测孔隙度")
        ax.set_ylabel("预测孔隙度")
        ax.set_title(f"RF 基线 R2={overall['r2']:.3f}")
        fig.tight_layout()
        png = os.path.join("reports", f"{args.dataset}_baseline_rf.png")
        fig.savefig(png, dpi=140)
        plt.close(fig)

    print(f"\n出图: {png}")
    js = os.path.join("reports", f"{args.dataset}_baseline.json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"已保存: {js}")


if __name__ == "__main__":
    main()
