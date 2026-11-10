"""构建带标签的窗口数据集。

用法:
    python build_dataset.py --dataset s1_demo
    python build_dataset.py --dataset demo01

换数据只改 configs/data.yaml，本文件无需修改。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from src.config import ConfigError, get_dataset, load_config
from src.io_wells import load_well
from src.windows import attach_labels, make_windows, summarize


def build(ds_name: str, cfg: dict):
    ds = get_dataset(cfg, ds_name)
    wcfg = ds["window"]
    length_m = float(wcfg["length_m"])
    overlap = float(wcfg["overlap"])

    print(f"== 数据集: {ds_name} ==")
    print(f"   窗口 {length_m} m / 重叠 {overlap:.0%} / 目标步长 {ds['target_step_m']} m")

    wells = []
    for wc in ds["wells"]:
        w = load_well(wc, ds)
        wells.append(w)
        print(
            f"   [{w.name}] {w.n_points} 点, {len(w.curves)} 通道, "
            f"标签 {len(w.labels) if w.labels else 0} 个"
            + (f" | {'; '.join(w.notes)}" if w.notes else "")
        )

    # 全局通道集合：按配置里的标准名顺序，取各井并集
    canonical = list(ds["channels"].keys())
    channels = [c for c in canonical if any(c in w.curves for w in wells)]
    if not channels:
        raise ConfigError("所有井都没有匹配到任何通道")

    # 岩性编码：全数据集统一
    lith_names = sorted(
        {
            str(x)
            for w in wells
            if w.labels is not None and w.labels.lith is not None
            for x in w.labels.lith
        }
    )
    lith_code = {n: i for i, n in enumerate(lith_names)}

    all_win, all_pres, all_y, all_nl, all_lith, all_lc, all_center, all_well = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    reports = []

    for w in wells:
        win, pres, starts, centers = make_windows(w, channels, length_m, overlap)
        y, n_label, lith, lith_consistent = attach_labels(starts, length_m, w.labels, lith_code)
        all_win.append(win)
        all_pres.append(pres)
        all_y.append(y)
        all_nl.append(n_label)
        all_lith.append(lith)
        all_lc.append(lith_consistent)
        all_center.append(centers)
        all_well.append(np.array([w.name] * win.shape[0]))
        reports.append(summarize(w.name, win, pres, y, n_label, lith_consistent))

    out = {
        "win": np.concatenate(all_win).astype(np.float32),
        "presence": np.concatenate(all_pres).astype(np.int8),
        "y": np.concatenate(all_y).astype(np.float32),
        "n_label": np.concatenate(all_nl).astype(np.int16),
        "lith": np.concatenate(all_lith).astype(np.int16),
        "lith_consistent": np.concatenate(all_lc).astype(np.int8),
        "depth_center": np.concatenate(all_center).astype(np.float32),
        "well": np.concatenate(all_well),
        "channels": np.array(channels),
        "lith_names": np.array(lith_names),
        "window_length_m": np.array([length_m]),
    }

    # ---- 打印 ---- #
    print("\n" + "=" * 78)
    print(
        f"{'井':<10}{'窗口':>6}{'有标签':>8}{'占比':>8}{'标签/窗':>9}"
        f"{'岩性一致':>9}{'y 均值':>9}{'y 标准差':>10}"
    )
    print("-" * 78)
    for r in reports:
        print(
            f"{r['well']:<10}{r['n_windows']:>6}{r['n_labeled']:>8}"
            f"{r['labeled_ratio']:>7.0%}{r['labels_per_window_mean']:>9.2f}"
            f"{(r['lith_consistent_ratio'] or 0):>8.0%}"
            f"{(r['y_mean'] or 0):>9.4f}{(r['y_std'] or 0):>10.4f}"
        )
    print("-" * 78)

    yv = out["y"][np.isfinite(out["y"])]
    total = out["win"].shape[0]
    labeled = int(np.isfinite(out["y"]).sum())
    print(
        f"合计: {total} 个窗口，其中 {labeled} 个有标签（{labeled / total:.0%}），"
        f"{total - labeled} 个无标签（留给自监督预训练）"
    )
    print(f"标签值域: {yv.min():.4f} – {yv.max():.4f}（小数孔隙度）")
    print(f"通道: {', '.join(channels)}")
    if lith_names:
        print(f"岩性: {', '.join(lith_names)}")

    # 留一井划分预览
    print("\n留一井划分预览：")
    for held in sorted({w.name for w in wells}):
        te = out["well"] == held
        tr = ~te
        print(
            f"  留 {held:<8} 训练标签 {int(np.isfinite(out['y'][tr]).sum()):>4} 个，"
            f"测试标签 {int(np.isfinite(out['y'][te]).sum()):>3} 个"
        )

    # ---- 存盘 ---- #
    proc = ds["paths"]["processed"]
    os.makedirs(proc, exist_ok=True)
    npz = os.path.join(proc, f"{ds_name}_labeled_windows.npz")
    np.savez_compressed(npz, **out)
    print(f"\n已保存: {npz}")

    os.makedirs("reports", exist_ok=True)
    rep = os.path.join("reports", f"{ds_name}_dataset_report.json")
    with open(rep, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": ds_name,
                "channels": channels,
                "lith_names": lith_names,
                "wells": reports,
                "total_windows": total,
                "labeled_windows": labeled,
            },
            f,
            ensure_ascii=False,
            indent=1,
        )
    print(f"已保存: {rep}")
    return out


def main():
    ap = argparse.ArgumentParser(description="构建带标签的窗口数据集")
    ap.add_argument("--dataset", required=True, help="configs/data.yaml 里 datasets 下的键名")
    ap.add_argument("--config", default=os.path.join("configs", "data.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    try:
        build(args.dataset, cfg)
    except (ConfigError, FileNotFoundError, ValueError, KeyError) as e:
        print(f"\n[配置/数据错误] {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
