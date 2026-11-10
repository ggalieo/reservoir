"""构建带标签的窗口数据集。

用法:
    python build_dataset.py --dataset force2020    # 分类（岩性）

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
    task = ds["task"]

    print(f"== 数据集: {ds_name}  任务: {task} ==")
    print(f"   窗口 {length_m} m / 重叠 {overlap:.0%} / 目标步长 {ds['target_step_m']} m")

    wells, skipped = [], []
    for wc in ds["wells"]:
        try:
            wells.append(load_well(wc, ds))
        except (ValueError, KeyError, FileNotFoundError) as e:
            skipped.append((wc.get("name", "?"), str(e).splitlines()[0]))
    if not wells:
        raise ValueError("没有任何一口井成功加载")

    print(f"   成功加载 {len(wells)} 口井" + (f"，跳过 {len(skipped)} 口" if skipped else ""))
    for name, why in skipped[:5]:
        print(f"     跳过 {name}: {why}")
    for w in wells[:3]:
        n_lab = len(w.labels) if w.labels else 0
        print(f"     [{w.name}] {w.n_points} 点, {len(w.curves)} 通道, 标签 {n_lab} 个")
    if len(wells) > 3:
        print(f"     ... 其余 {len(wells) - 3} 口略")

    canonical = list(ds["channels"].keys())
    channels = [c for c in canonical if any(c in w.curves for w in wells)]
    if not channels:
        raise ConfigError("所有井都没有匹配到任何通道")

    lith_names = sorted(
        {
            str(x)
            for w in wells
            if w.labels is not None and w.labels.lith is not None
            for x in w.labels.lith
        }
    )
    lith_code = {n: i for i, n in enumerate(lith_names)}

    acc = {k: [] for k in ("win", "pres", "y", "nl", "lith", "lc", "center", "well")}
    reports = []

    for w in wells:
        win, pres, starts, centers = make_windows(w, channels, length_m, overlap)
        y, n_label, lith, lith_consistent = attach_labels(
            starts, length_m, w.labels, lith_code, mode=task
        )
        acc["win"].append(win)
        acc["pres"].append(pres)
        acc["y"].append(y)
        acc["nl"].append(n_label)
        acc["lith"].append(lith)
        acc["lc"].append(lith_consistent)
        acc["center"].append(centers)
        acc["well"].append(np.array([w.name] * win.shape[0]))
        reports.append(summarize(w.name, win, pres, y, n_label, lith_consistent))

    out = {
        "win": np.concatenate(acc["win"]).astype(np.float32),
        "presence": np.concatenate(acc["pres"]).astype(np.int8),
        "y": np.concatenate(acc["y"]).astype(np.float32),
        "n_label": np.concatenate(acc["nl"]).astype(np.int16),
        "lith": np.concatenate(acc["lith"]).astype(np.int16),
        "lith_consistent": np.concatenate(acc["lc"]).astype(np.int8),
        "depth_center": np.concatenate(acc["center"]).astype(np.float32),
        "well": np.concatenate(acc["well"]),
        "channels": np.array(channels),
        "lith_names": np.array(lith_names),
        "window_length_m": np.array([length_m]),
        "task": np.array([task]),
    }

    labeled = int(np.isfinite(out["y"]).sum())
    total = out["win"].shape[0]

    if task == "classification":
        yv = out["y"][np.isfinite(out["y"])].astype(int)
        class_ids = np.unique(yv)
        out["class_ids"] = class_ids.astype(np.int32)
        print(
            f"\n合计 {total} 窗口，有标签 {labeled}（{labeled / total:.0%}），"
            f"{len(class_ids)} 个类别"
        )
        print(f"\n{'类别码':>8}{'窗口数':>8}{'占比':>8}")
        for c in class_ids:
            n = int((yv == c).sum())
            print(f"{c:>8}{n:>8}{n / labeled:>8.1%}")
    else:
        yv = out["y"][np.isfinite(out["y"])]
        print(
            f"\n合计 {total} 窗口，有标签 {labeled}（{labeled / total:.0%}），"
            f"{total - labeled} 个无标签（留给自监督预训练）"
        )
        print(f"标签值域 {yv.min():.4f} – {yv.max():.4f}，均值 {yv.mean():.4f}")
        out["class_ids"] = np.array([], dtype=np.int32)

    print(f"通道: {', '.join(channels)}")
    print("\n留一井划分预览（前 5 折）:")
    wnames = sorted({w.name for w in wells})
    for held in wnames[:5]:
        te = out["well"] == held
        print(
            f"  留 {held:<16} 训练 {int(np.isfinite(out['y'][~te]).sum()):>7} / "
            f"测试 {int(np.isfinite(out['y'][te]).sum()):>6}"
        )

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
                "task": task,
                "channels": channels,
                "n_wells": len(wells),
                "skipped": skipped,
                "total_windows": total,
                "labeled_windows": labeled,
                "class_ids": out["class_ids"].tolist(),
            },
            f,
            ensure_ascii=False,
            indent=1,
        )
    print(f"已保存: {rep}")
    return out


def main():
    ap = argparse.ArgumentParser(description="构建带标签的窗口数据集")
    ap.add_argument("--dataset", required=True)
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
