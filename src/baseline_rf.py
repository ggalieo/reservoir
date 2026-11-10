"""RandomForest 基线（B1）：窗口统计量 -> 孔隙度。

目的不是刷精度，而是给出一个锚点：
    - 简单模型在这套数据上能做到多少
    - 后续表征模型要证明的是"在缺失/错位下更稳"，而不是"精度更高"

划分：留一井（LOO），同一口井的窗口不会同时出现在训练和测试。
用法：python -m src.baseline_rf --dataset s1_demo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

STATS = ("mean", "std", "min", "max", "p10", "p90")


def window_features(win: np.ndarray, channels: list[str]):
    """每个通道取若干统计量，拼成 (N, C*len(STATS))。"""
    n, c, _ = win.shape
    cols, names = [], []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for ci, ch in enumerate(channels):
            b = win[:, ci, :].astype(np.float64)
            cols.append(np.nanmean(b, axis=1))
            cols.append(np.nanstd(b, axis=1))
            cols.append(np.nanmin(b, axis=1))
            cols.append(np.nanmax(b, axis=1))
            cols.append(np.nanpercentile(b, 10, axis=1))
            cols.append(np.nanpercentile(b, 90, axis=1))
            names += [f"{ch}_{s}" for s in STATS]

    X = np.column_stack(cols)
    return X, names


def impute_train_only(Xtr, Xte):
    """用训练集的中位数填补 NaN，避免测试集信息泄漏。"""
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xtr = np.where(np.isfinite(Xtr), Xtr, med)
    Xte = np.where(np.isfinite(Xte), Xte, med)
    return Xtr, Xte, med


def metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "n": int(len(y_true))}


def run(dataset: str, n_estimators: int, seeds: int):
    path = os.path.join("data", "processed", f"{dataset}_labeled_windows.npz")
    if not os.path.exists(path):
        sys.exit(f"找不到数据集 {path}\n先运行: python build_dataset.py --dataset {dataset}")

    d = np.load(path, allow_pickle=True)
    win, y, wells = d["win"], d["y"], d["well"].astype(str)
    lith = d["lith"] if "lith" in d.files else np.full(y.shape, -1, dtype=np.int16)
    lith_names = d["lith_names"].tolist() if d["lith_names"].size else []
    channels = d["channels"].tolist()

    # 过滤：必须有标签 + 至少一个通道有观测（排除全 NaN 空窗）
    has_label = np.isfinite(y)
    has_data = np.isfinite(win).any(axis=(1, 2))
    mask = has_label & has_data
    dropped_label = int((~has_label).sum())
    dropped_empty = int((has_label & ~has_data).sum())

    win, y, wells = win[mask], y[mask], wells[mask]
    lith = lith[mask]
    print(f"== {dataset} ==")
    print(
        f"   总窗口 {mask.size}，无标签 {dropped_label}，空窗 {dropped_empty} → 参与建模 {mask.sum()}"
    )

    X, feat_names = window_features(win, channels)
    print(f"   特征 {X.shape[1]} 维（{len(channels)} 通道 × {len(STATS)} 统计量）")

    uniq_wells = sorted(set(wells.tolist()))
    if len(uniq_wells) < 2:
        sys.exit(f"只有 {len(uniq_wells)} 口井，无法做留一井。至少需要 2 口。")
    print(f"   留一井划分: {', '.join(uniq_wells)}")

    folds, pooled_pred, pooled_true, pooled_well, pooled_lith = [], [], [], [], []
    importances = []

    for seed in range(seeds):
        for held in uniq_wells:
            te = wells == held
            tr = ~te
            Xtr, Xte, _ = impute_train_only(X[tr], X[te])

            # 平凡基线：永远预测训练集均值
            trivial = np.full(int(te.sum()), float(np.mean(y[tr])))

            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                min_samples_leaf=3,
                random_state=seed,
                n_jobs=-1,
            )
            rf.fit(Xtr, y[tr])
            pred = rf.predict(Xte)

            m = metrics(y[te], pred)
            m_trivial = metrics(y[te], trivial)
            m.update(
                {
                    "well": held,
                    "seed": seed,
                    "trivial_rmse": m_trivial["rmse"],
                    "trivial_r2": m_trivial["r2"],
                }
            )
            folds.append(m)

            pooled_pred.append(pred)
            pooled_true.append(y[te])
            pooled_well.append(wells[te])
            pooled_lith.append(lith[te])
            importances.append(rf.feature_importances_)

    # ---- 汇总 ---- #
    print("\n" + "=" * 74)
    print(f"{'井':<8}{'n':>5}{'RMSE':>9}{'MAE':>9}{'R2':>9}{'均值基线RMSE':>14}{'提升':>8}")
    print("-" * 74)
    by = {}
    for m in folds:
        by.setdefault(m["well"], []).append(m)
    for w in uniq_wells:
        ms = by[w]
        n = int(np.mean([m["n"] for m in ms]))
        rmse = float(np.mean([m["rmse"] for m in ms]))
        mae = float(np.mean([m["mae"] for m in ms]))
        r2 = float(np.mean([m["r2"] for m in ms]))
        tr_rmse = float(np.mean([m["trivial_rmse"] for m in ms]))
        gain = (tr_rmse - rmse) / tr_rmse * 100
        print(f"{w:<8}{n:>5}{rmse:>9.4f}{mae:>9.4f}{r2:>9.3f}{tr_rmse:>14.4f}{gain:>7.0f}%")
    print("-" * 74)

    pt, pp = np.concatenate(pooled_true), np.concatenate(pooled_pred)
    overall = metrics(pt, pp)
    print(
        f"{'合计':<8}{overall['n']:>5}{overall['rmse']:>9.4f}{overall['mae']:>9.4f}"
        f"{overall['r2']:>9.3f}"
    )

    # ---- 分岩性统计：检验"R2 是否主要来自区分岩性" ---- #
    pl = np.concatenate(pooled_lith)
    lith_stats = {}
    if lith_names:
        print(f"\n{'岩性':<10}{'n':>6}{'实测均值':>10}{'实测标准差':>12}{'RMSE':>9}{'R2':>9}")
        print("-" * 58)
        for code, nm in enumerate(lith_names):
            m = pl == code
            if m.sum() < 5:
                continue
            yy, yp = pt[m], pp[m]
            mm = metrics(yy, yp)
            lith_stats[nm] = {
                "n": int(m.sum()),
                "y_mean": float(yy.mean()),
                "y_std": float(yy.std()),
                "rmse": mm["rmse"],
                "r2": mm["r2"],
            }
            print(
                f"{nm:<10}{mm['n']:>6}{yy.mean():>10.4f}{yy.std():>12.4f}"
                f"{mm['rmse']:>9.4f}{mm['r2']:>9.3f}"
            )
    within = [v["r2"] for v in lith_stats.values() if np.isfinite(v["r2"])]
    if within:
        print("-" * 58)
        print(f"岩性内平均 R2 = {np.mean(within):.3f}（总体 {overall['r2']:.3f}）")
        print("若明显低于总体，说明总体精度主要来自区分岩性，而非岩性内的精细预测。")

    r2s = [m["r2"] for m in folds]
    print(
        f"\nR2 跨折稳定性: 均值 {np.mean(r2s):.3f}，标准差 {np.std(r2s):.3f}，"
        f"范围 {np.min(r2s):.3f} – {np.max(r2s):.3f}"
    )

    imp = np.mean(importances, axis=0)
    order = np.argsort(imp)[::-1][:10]
    print("\n特征重要性 Top 10：")
    for i in order:
        print(f"   {feat_names[i]:<14}{imp[i]:.3f}")

    # ---- 出图 ---- #
    os.makedirs("reports", exist_ok=True)
    png = os.path.join("reports", f"{dataset}_baseline_rf.png")
    fig, ax = plt.subplots(figsize=(6.5, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(uniq_wells)))
    for c, w in zip(colors, uniq_wells):
        m = np.concatenate(pooled_well) == w
        ax.scatter(pt[m], pp[m], s=18, alpha=0.75, color=c, label=f"留出 {w}")
    lim = [float(min(pt.min(), pp.min())) - 0.01, float(max(pt.max(), pp.max())) + 0.01]
    ax.plot(lim, lim, "k--", lw=1, label="1:1")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("实测孔隙度")
    ax.set_ylabel("预测孔隙度")
    ax.set_title(f"RF 基线（留一井）  R2={overall['r2']:.3f}  RMSE={overall['rmse']:.4f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"\n出图: {png}")

    out = {
        "dataset": dataset,
        "n_samples": int(mask.sum()),
        "dropped_no_label": dropped_label,
        "dropped_empty_window": dropped_empty,
        "channels": channels,
        "n_features": int(X.shape[1]),
        "n_estimators": n_estimators,
        "seeds": seeds,
        "folds": folds,
        "overall": overall,
        "lith_stats": lith_stats,
        "r2_within_lith_mean": float(np.mean(within)) if within else None,
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "feature_importance_top10": [
            {"feature": feat_names[i], "importance": float(imp[i])} for i in order
        ],
    }
    js = os.path.join("reports", f"{dataset}_baseline_rf.json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"已保存: {js}")
    return out


def main():
    ap = argparse.ArgumentParser(description="RandomForest 基线")
    ap.add_argument("--dataset", default="s1_demo")
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    run(args.dataset, args.n_estimators, args.seeds)


if __name__ == "__main__":
    main()
