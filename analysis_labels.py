"""接真实数据后的两项诊断：

1. 岩心标签能否与 LAS 深度对齐（匹配率、最近邻距离、按岩性统计）
2. RT 的"尖峰"是真异常还是地层信号（连通段长度分析）

用法: python analysis_labels.py
"""
import lasio
import numpy as np
import pandas as pd

LAS = "data/raw/S1.las"
CORE = "data/raw/core_porosity.csv"


def check_core():
    print("=" * 62)
    print("① 岩心标签对齐检查")
    print("=" * 62)

    las = lasio.read(LAS)
    depth = np.asarray(las.index, dtype=float)

    core = pd.read_csv(CORE)
    c1 = core[core["井名"] == "S1"].copy()
    print(f"CSV 总样品 {len(core)} 个，其中 S1 {len(c1)} 个")
    print(f"LAS 深度 {depth[0]:.2f} – {depth[-1]:.2f} m，步长 {np.median(np.diff(depth)):.3f} m")

    # 最近邻匹配
    idx = np.searchsorted(depth, c1["深度(m)"].values)
    idx = np.clip(idx, 0, depth.size - 1)
    dist = np.abs(depth[idx] - c1["深度(m)"].values)
    c1["匹配深度"] = depth[idx]
    c1["偏差(m)"] = dist

    print(f"\n匹配成功率（偏差 ≤ 0.0625 m，即半个采样点）: "
          f"{(dist <= 0.0625).mean():.1%}")
    print(f"最近邻偏差: 中位数 {np.median(dist):.3f} m, 最大 {dist.max():.3f} m")

    print("\n按岩性统计孔隙度 (%)：")
    g = c1.groupby("岩性")["孔隙度(%)"].agg(["count", "mean", "std", "min", "max"]).round(2)
    print(g.to_string())

    print("\n按岩性统计渗透率 (mD)：")
    g2 = c1.groupby("岩性")["渗透率(mD)"].agg(["count", "mean", "min", "max"]).round(3)
    print(g2.to_string())

    print(f"\n标签值域: 孔隙度 {c1['孔隙度(%)'].min():.2f} – {c1['孔隙度(%)'].max():.2f} %")
    return c1


def check_rt_spikes():
    print("\n" + "=" * 62)
    print("② RT 尖峰性质检查（是仪器故障还是地层信号？）")
    print("=" * 62)

    las = lasio.read(LAS)
    depth = np.asarray(las.index, dtype=float)
    rt = np.asarray(las["RT"].data, dtype=float)
    rt = np.where(rt <= -900, np.nan, rt)

    med = np.nanmedian(rt)
    mad = np.nanmedian(np.abs(rt - med))
    z = np.abs(rt - med) / (1.4826 * mad)
    flag = z > 8

    print(f"RT 中位数 {med:.2f} Ω·m，MAD {mad:.3f}")
    print(f"被标记 {int(np.nansum(flag))} 点（{(flag & np.isfinite(rt)).sum() / np.isfinite(rt).sum():.1%}）")

    # 连通段分析：真实地层异常会形成长段，仪器尖峰是孤立点
    runs = []
    cur = 0
    for f in np.r_[flag, False]:
        if f:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    runs = np.array(runs)
    if runs.size:
        print(f"连通段数 {runs.size}，长度分布: 最长 {runs.max()} 点 "
              f"({runs.max() * 0.125:.1f} m)，中位 {int(np.median(runs))} 点")
        print(f"孤立点（长度 ≤ 2）占比: {(runs <= 2).mean():.1%}")
        long_runs = (runs > 10).sum()
        print(f"长段（> 10 点 = 1.25 m）数: {long_runs}")

        # 找到最长段的深度范围
        i = 0
        best, best_i = 0, 0
        for j, f in enumerate(np.r_[flag, False]):
            if f:
                i += 1
            else:
                if i > best:
                    best, best_i = i, j - i
                i = 0
        print(f"最长段深度: {depth[best_i]:.1f} – {depth[best_i + best - 1]:.1f} m，"
              f"该段 RT 值 {np.nanmin(rt[best_i:best_i + best]):.0f} – "
              f"{np.nanmax(rt[best_i:best_i + best]):.0f} Ω·m")

    print("\n结论：若长段占比高，说明标记的是地层（灰岩高阻），"
          "稳健 z 在双峰分布上不适用，需按层段或按岩性分别质控。")


if __name__ == "__main__":
    check_core()
    check_rt_spikes()
