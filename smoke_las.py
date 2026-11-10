"""LAS 格式的井数据检查：读入 → 质控 → 出图 → 切窗口 → 存盘。

用于 LAS 格式（FORCE 2020 走 CSV 路径，见 build_dataset.py）。

这是"第一步"的验收脚本。它不碰任何模型，只回答三个问题：
    1. 这份 LAS 能不能读进来？有哪些曲线、采样多少、深度范围多少？
    2. 曲线质量如何？缺失、尖峰、井径异常各在哪一段？
    3. 按 2 m / 50% 重叠切窗，能得到多少个窗口、缺失率多少？

用法:
    python smoke_las.py data/raw/<井名>.las
"""

import argparse
import os
import sys

import lasio
import matplotlib

matplotlib.use("Agg")  # 无界面后端，服务器上也能跑
import matplotlib.pyplot as plt
import numpy as np

# 让中文标题正常显示
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 目标通道：按 mnemonic 模糊匹配，兼容不同公司的命名习惯
CHANNELS = {
    "GR": ("GR", "GAM", "GAPI", "GRD"),
    "SP": ("SP", "SPC"),
    "CAL": ("CAL", "CALI", "CALIPER"),
    "RT": ("RT", "ILD", "RILD", "LLD", "RDEEP", "RES"),
    "DEN": ("DEN", "RHOB", "ZDEN"),
    "CNU": ("CNU", "CNL", "NPHI", "TNPH"),
    "AC": ("AC", "DT", "DTC", "SONIC"),
}


def match_channels(las):
    """返回 {目标名: 实际 mnemonic}。"""
    have = {c.mnemonic.upper(): c.mnemonic for c in las.curves}
    picked = {}
    for std, cands in CHANNELS.items():
        for cand in cands:
            hit = next((v for k, v in have.items() if k == cand), None)
            if hit is None:
                hit = next((v for k, v in have.items() if k.startswith(cand)), None)
            if hit:
                picked[std] = hit
                break
    return picked


def qc_report(depth, curves):
    """逐通道统计：缺失率、尖峰、井径异常。"""
    rows = []
    for name, arr in curves.items():
        n = arr.size
        nan = int(np.isnan(arr).sum())
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            rows.append((name, nan / n, 0, "-"))
            continue
        med = np.median(valid)
        mad = np.median(np.abs(valid - med)) or 1e-9
        robust_z = np.abs(valid - med) / (1.4826 * mad)
        spikes = int((robust_z > 8).sum())
        rows.append((name, nan / n, spikes, f"{valid.min():.3g} ~ {valid.max():.3g}"))
    return rows


def make_windows(depth, curves, window_m=2.0, overlap=0.5):
    """按深度窗口切分，返回 (N, C, L) 与 presence(N, C)。

    缺失通道不补零：整段缺失的通道 presence=0，局部缺失保留 NaN 让上层决定。
    """
    names = list(curves)
    step = float(np.median(np.diff(depth)))
    L = int(round(window_m / step))
    stride = max(1, int(round(L * (1 - overlap))))
    n_win = max(0, (depth.size - L) // stride + 1)

    arr = np.stack([curves[k] for k in names], axis=0)  # (C, N)
    win = np.empty((n_win, arr.shape[0], L), dtype=np.float32)
    presence = np.ones((n_win, arr.shape[0]), dtype=np.int8)
    starts = np.empty(n_win, dtype=np.float32)

    for i in range(n_win):
        s = i * stride
        block = arr[:, s : s + L]
        win[i] = block
        starts[i] = depth[s]
        # 整窗全缺 → presence=0
        presence[i] = (np.isfinite(block).sum(axis=1) > 0).astype(np.int8)

    return names, win, presence, starts


def plot(depth, curves, core, gap_mask, out_png, well):
    tracks = [(k, k) for k in ("GR", "RT", "DEN", "AC") if k in curves]
    if not tracks:
        tracks = [(k, k) for k in list(curves)[:4]]

    fig, axes = plt.subplots(
        1,
        len(tracks) + 1,
        figsize=(3.0 * (len(tracks) + 1), 8),
        sharey=True,
        gridspec_kw={"width_ratios": [1] * len(tracks) + [0.7]},
    )

    for ax, (name, _) in zip(axes[:-1], tracks):
        ax.plot(curves[name], depth, lw=0.8, color="#1f4e79")
        ax.set_title(name)
        ax.grid(alpha=0.3)
        ax.invert_yaxis()
        # 缺失段涂灰
        if gap_mask.any():
            ax.fill_betweenx(
                depth,
                np.nanmin(curves[name]),
                np.nanmax(curves[name]),
                where=gap_mask,
                color="0.85",
                zorder=0,
            )
    axes[0].set_ylabel("深度 (m)")

    ax = axes[-1]
    ax.plot(core, depth, "o", ms=4, color="#c00000")
    ax.set_title("岩心孔隙度")
    ax.grid(alpha=0.3)
    fig.suptitle(f"{well}　冒烟测试：曲线质控与窗口化", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("las_path", help="LAS 文件路径（必填）")
    ap.add_argument("--window", type=float, default=2.0, help="窗口长度 (m)")
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--outdir", default="reports")
    args = ap.parse_args()

    if not os.path.exists(args.las_path):
        sys.exit(f"找不到文件: {args.las_path}")

    las = lasio.read(args.las_path)
    well = str(las.well.WELL.value or os.path.basename(args.las_path))
    depth = np.asarray(las.index, dtype=float)

    print(f"== {well} ==")
    print(
        f"深度 {depth[0]:.2f} – {depth[-1]:.2f} m，{depth.size} 点，"
        f"步长 {np.median(np.diff(depth)):.3f} m"
    )
    print(f"曲线总数 {len(las.curves) - 1}：", ", ".join(c.mnemonic for c in las.curves[1:]))

    picked = match_channels(las)
    print(f"\n匹配到标准通道 {len(picked)}/{len(CHANNELS)}：")
    for k, v in picked.items():
        print(f"  {k:<4} <- {v}")

    if len(picked) < 3:
        sys.exit("\n可用通道不足 3 条，无法继续。检查曲线命名或换一口井。")

    curves = {std: np.asarray(las[m].data, dtype=float) for std, m in picked.items()}
    curves = {k: np.where(v <= -900, np.nan, v) for k, v in curves.items()}  # 常见 NULL 值

    print("\n质量报告：")
    print(f"  {'通道':<6}{'缺失率':>8}{'尖峰数':>8}   值域")
    for name, miss, spikes, rng in qc_report(depth, curves):
        print(f"  {name:<6}{miss:>8.1%}{spikes:>8}   {rng}")

    core_m = next(
        (c.mnemonic for c in las.curves if c.mnemonic.upper() in ("CPOR", "CORE", "PORO")), None
    )
    core = np.asarray(las[core_m].data, dtype=float) if core_m else np.full(depth.size, np.nan)
    core = np.where(core <= -900, np.nan, core)
    print(
        f"\n岩心标签：{int(np.isfinite(core).sum())} 个点"
        + (f"（列 {core_m}）" if core_m else "（无岩心列）")
    )

    gap_mask = np.zeros(depth.size, dtype=bool)
    for arr in curves.values():
        gap_mask |= ~np.isfinite(arr)

    names, win, presence, starts = make_windows(depth, curves, args.window, args.overlap)
    print(f"\n窗口化：{args.window} m / 重叠 {args.overlap:.0%}")
    print(f"  窗口数 {win.shape[0]}，形状 {win.shape}（窗口 × 通道 × 深度点）")
    print(
        "  通道存在率："
        + "，".join(f"{n}={presence[:, i].mean():.0%}" for i, n in enumerate(names))
    )
    print(f"  窗口内缺失采样占比 {np.isnan(win).mean():.1%}")

    os.makedirs(args.outdir, exist_ok=True)
    png = os.path.join(args.outdir, f"smoke_{well}.png")
    plot(depth, curves, core, gap_mask, png, well)
    print(f"\n出图: {png}")

    npz = os.path.join("data", "processed", f"{well}_win.npz")
    os.makedirs(os.path.dirname(npz), exist_ok=True)
    np.savez_compressed(npz, win=win, presence=presence, starts=starts, channels=np.array(names))
    print(f"窗口存盘: {npz}")

    print("\n[通过] 数据链路可用：读入 → 质控 → 窗口化 → 落盘")


if __name__ == "__main__":
    main()
