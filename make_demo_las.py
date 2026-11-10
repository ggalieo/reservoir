"""生成一口合成井的 LAS 文件，用于在拿到真实数据前验证整条数据链路。

刻意做得"像真的"：包含深度缺失段、异常尖峰、以及孔渗饱的岩心标签列，
这样冒烟测试能同时验证质控逻辑与标签对齐逻辑。

用法:
    python make_demo_las.py                 # 写到 data/raw/DEMO-01.las
    python make_demo_las.py --well W-07 --out data/raw/W-07.las
"""

import argparse
import os

import numpy as np
from lasio import LASFile

CURVES = {
    # mnemonic: (unit, 描述, 典型范围, 噪声)
    "GR": ("API", "Gamma Ray", (30, 120), 6.0),
    "SP": ("mV", "Spontaneous Potential", (-40, 10), 2.0),
    "CAL": ("in", "Caliper", (8.4, 9.2), 0.15),
    "RT": ("ohm.m", "Deep Resistivity", (2, 60), 0.3),
    "DEN": ("g/cm3", "Bulk Density", (2.2, 2.6), 0.02),
    "CNU": ("v/v", "Neutron Porosity", (0.08, 0.35), 0.01),
    "AC": ("us/ft", "Sonic Transit Time", (55, 95), 2.0),
}


def build(well="DEMO-01", top=2100.0, bot=2160.0, step=0.125, seed=0):
    rng = np.random.default_rng(seed)
    depth = np.arange(top, bot, step)

    # 分层：三层，各自有不同的岩石物性基底，模拟真实层序
    layer = np.digitize(depth, [top + 15, top + 38])
    base = {
        0: dict(poro=0.22, gr=55, den=2.38, ac=78),  # 泥质偏重
        1: dict(poro=0.13, gr=85, den=2.50, ac=88),  # 致密泥岩
        2: dict(poro=0.18, gr=42, den=2.30, ac=70),  # 较优储层
    }

    n = depth.size
    poro_true = np.array([base[k]["poro"] for k in layer])
    poro_true = poro_true + rng.normal(0, 0.015, n)

    data = {}
    data["GR"] = np.array([base[k]["gr"] for k in layer]) + rng.normal(0, 6, n)
    data["SP"] = -30 + 40 * (0.25 - poro_true) / 0.25 + rng.normal(0, 2, n)
    data["CAL"] = 8.7 + rng.normal(0, 0.15, n)
    data["DEN"] = (
        np.array([base[k]["den"] for k in layer])
        - 0.6 * (poro_true - 0.15)
        + rng.normal(0, 0.02, n)
    )
    data["CNU"] = poro_true * 0.9 + rng.normal(0, 0.01, n)
    data["AC"] = (
        np.array([base[k]["ac"] for k in layer]) + 120 * (poro_true - 0.15) + rng.normal(0, 2, n)
    )
    data["RT"] = 5 + 30 * (0.25 - poro_true) / 0.25 + rng.normal(0, 0.3, n)

    # 人为制造异常：井径突变段（垮塌）与仪器尖峰
    cal_bad = (depth > top + 22) & (depth < top + 24)
    data["CAL"][cal_bad] += 3.0
    spike = rng.choice(n, size=5, replace=False)
    data["RT"][spike] *= 50

    # 人为制造缺失段：录井/曲线中断 1.5 m
    gap = (depth > top + 45) & (depth < top + 46.5)
    for k in ("GR", "DEN", "AC"):
        data[k][gap] = np.nan

    return depth, data, poro_true, gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--well", default="DEMO-01")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    depth, data, poro_true, gap = build(args.well)
    las = LASFile()
    las.well.WELL = args.well
    las.well.FLD = "SYNTHETIC"
    las.well.SRVC = "demo-generator"
    las.well.DATE = "2026-11-07"
    las.well.STRT = float(depth[0])
    las.well.STOP = float(depth[-1])
    las.well.STEP = 0.125
    las.well.NULL = -999.25

    las.append_curve("DEPT", depth, unit="m", descr="Depth")
    for mnem, (unit, desc, _, _) in CURVES.items():
        las.append_curve(mnem, data[mnem], unit=unit, descr=desc)
    # 岩心标签列：只有稀疏采样点有值（模拟取心段）
    core = np.full(depth.size, np.nan)
    idx = np.arange(0, depth.size, 40)  # 每 5 m 一个点
    core[idx] = poro_true[idx]
    las.append_curve("CPOR", core, unit="v/v", descr="Core Porosity")

    out = args.out or os.path.join("data", "raw", f"{args.well}.las")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    las.write(out, version=2.0)

    n_missing = int(np.isnan(data["GR"]).sum())
    print(f"wrote {out}")
    print(f"  depth {depth[0]:.2f}-{depth[-1]:.2f} m, {depth.size} points, step 0.125 m")
    print(f"  curves: {', '.join(CURVES)}")
    print(f"  core labels: {int(np.isfinite(core).sum())} points")
    print(f"  injected gap: {int(gap.sum())} samples ({n_missing} NaN in GR)")


if __name__ == "__main__":
    main()
