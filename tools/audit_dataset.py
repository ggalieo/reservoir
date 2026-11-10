"""数据集审计：检查遗漏、逻辑错误与未覆盖的边界情况。

只读不改，输出问题清单。
用法: python tools/audit_dataset.py --dataset force2020
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baseline_rf import make_folds  # noqa: E402

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn

    return deco


def load(dataset):
    p = os.path.join("data", "processed", f"{dataset}_labeled_windows.npz")
    if not os.path.exists(p):
        sys.exit(f"找不到 {p}")
    return np.load(p, allow_pickle=True)


@check("1. 标签完整性")
def check_labels(d):
    y = d["y"]
    n = y.size
    nan = int(np.isnan(y).sum())
    zero = int((y == 0).sum())
    print(f"  窗口 {n:,}，无标签 {nan:,}（{nan / n:.2%}），y==0 的 {zero}")
    if zero:
        print("  [!!] y==0 可能是'无标签被填 0'，需确认")
    if nan == n:
        print("  [!!] 全部无标签")


@check("2. 空窗口（全通道无观测）")
def check_empty(d):
    win = d["win"]
    empty = ~np.isfinite(win).any(axis=(1, 2))
    print(f"  空窗口 {int(empty.sum())} / {win.shape[0]}")
    if empty.sum():
        wells = d["well"][empty]
        print(f"  涉及井: {sorted(set(wells.tolist()))[:5]}")


@check("3. presence 是二值，但覆盖率可能很低（逻辑缺口）")
def check_coverage(d):
    win, presence = d["win"], d["presence"]
    fin = np.isfinite(win)
    cov = fin.mean(axis=2)  # (N, C)
    C = win.shape[1]
    chans = d["channels"].tolist()
    bad = 0
    for c in range(C):
        m = presence[:, c] == 1
        if m.sum() == 0:
            print(f"  {chans[c]:<4} 从未出现")
            continue
        low = (cov[m, c] < 0.2).sum()
        bad += low
        print(
            f"  {chans[c]:<4} presence=1 的窗口 {int(m.sum()):>7}，"
            f"其中覆盖率<20% 的 {low:>5}，平均覆盖率 {cov[m, c].mean():.1%}"
        )
    if bad:
        print(f"  [!!] 共 {bad} 个窗口 presence=1 但覆盖率<20%：二值标记会高估'有该模态'")


@check("4. 按井分组划分是否泄漏")
def check_split(d):
    wells = d["well"].astype(str)
    folds = make_folds(wells, 5)
    ok = True
    for name, tr, te in folds:
        overlap = set(wells[tr].tolist()) & set(wells[te].tolist())
        if overlap:
            ok = False
            print(f"  [!!] {name} 训练/测试共享井: {list(overlap)[:3]}")
    print(f"  5 折，井级无交集: {ok}")


@check("5. 各折的类别覆盖（稀有类可能在某折完全缺失）")
def check_fold_classes(d):
    y, wells = d["y"], d["well"].astype(str)
    m = np.isfinite(y)
    y, wells = y[m].astype(int), wells[m]
    folds = make_folds(wells, 5)
    classes = np.unique(y)
    print(f"  类别数 {len(classes)}")
    for name, tr, te in folds:
        te_c = set(y[te].tolist())
        tr_c = set(y[tr].tolist())
        missing_te = [c for c in classes if c not in te_c]
        missing_tr = [c for c in classes if c not in tr_c]
        print(
            f"  {name:<22} 测试缺类 {len(missing_te):>2}，训练缺类 {len(missing_tr):>2}"
            + (f"  [测试缺 {missing_te[:3]}]" if missing_te else "")
        )
    tot_missing = sum(1 for _, _, te in folds for c in classes if c not in set(y[te].tolist()))
    if tot_missing:
        print(f"  [!!] 共 {tot_missing} 个（折,类）测试集中该类缺失 → 该折对这类无评估能力")


@check("6. 特征矩阵：恒定列与残余 NaN")
def check_features(d):
    from src.baseline_rf import window_features

    win = d["win"]
    mask = np.isfinite(d["y"]) & np.isfinite(win).any(axis=(1, 2))
    X, names = window_features(win[mask], d["channels"].tolist())
    nan = np.isnan(X)
    print(f"  特征矩阵 {X.shape}，NaN 元素 {int(nan.sum()):,}（{nan.mean():.2%}）")
    col_nan = nan.mean(axis=0)
    worst = np.argsort(col_nan)[::-1][:3]
    for i in worst:
        if col_nan[i] > 0:
            print(f"    {names[i]:<12} NaN {col_nan[i]:.1%}")
    const = np.nanstd(X, axis=0) < 1e-12
    print(
        f"  训练集常数/退化列: {int(const.sum())}"
        + (f" → {[names[i] for i in np.where(const)[0]][:5]}" if const.sum() else "")
    )


@check("7. 曲线值域 vs NULL 判据（<=-900 会被当缺失）")
def check_null_heuristic(d):
    win, chans = d["win"], d["channels"].tolist()
    for c, name in enumerate(chans):
        v = win[:, c, :]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        print(f"  {name:<4} 最小 {v.min():>10.3f}  最大 {v.max():>10.3f}", end="")
        if v.min() <= -900:
            print("  [!!] 触到 -900 判据，可能误删真实值")
        else:
            print()


@check("8. 窗口重叠导致的样本重复（同一深度被多个窗口覆盖）")
def check_overlap(d, dataset):
    wl = float(d["window_length_m"][0])
    dc = d["depth_center"]
    wells = d["well"].astype(str)
    w0 = wells[0]
    d0 = np.sort(dc[wells == w0])
    if d0.size > 2:
        step = np.median(np.diff(d0))
        print(f"  窗口长度 {wl} m，中心间距 {step:.3f} m → 每个深度被约 {wl / step:.1f} 个窗口覆盖")
        if wl / step > 1.5:
            print("  [注] 重叠窗口共享深度，测试集样本不独立；这是设计选择，但需在论文中说明")


@check("9. 深度单调性与重复深度")
def check_depth(d):
    dc, wells = d["depth_center"], d["well"].astype(str)
    bad = 0
    for w in sorted(set(wells.tolist()))[:200]:
        d0 = dc[wells == w]
        if d0.size < 2:
            continue
        if not np.all(np.diff(d0) > 0):
            bad += 1
    print(f"  检查前 200 口井，深度非严格递增的井: {bad}")


@check("10. 类分布与随机基线")
def check_baseline_floor(d):
    y = d["y"]
    yv = y[np.isfinite(y)].astype(int)
    vals, counts = np.unique(yv, return_counts=True)
    p = counts / counts.sum()
    print(f"  众数类占比 {p.max():.1%} → 全预测众数的 acc 下限约 {p.max():.3f}")
    print(f"  均衡随机（按先验猜）的 acc 约 {np.sum(p**2):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="force2020")
    args = ap.parse_args()

    d = load(args.dataset)
    print(f"=== 审计 {args.dataset} ===")
    print(f"任务: {d['task'][0] if 'task' in d.files else '?'}")
    for name, fn in CHECKS:
        print(f"\n{name}")
        try:
            if fn.__code__.co_argcount == 2:
                fn(d, args.dataset)
            else:
                fn(d)
        except Exception as e:
            print(f"  [检查失败] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
