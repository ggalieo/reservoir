"""把 FORCE 2020 的 train.csv 转成"每井一个 CSV + 一个标签表"。

为什么转 CSV 而不是 LAS：
    FORCE 的深度采样不是等间距（0.12~0.16 m 不等），写 LAS 需要伪造步长，
    而 CSV 保留原始采样，由下游统一重采样。真实数据导出成 CSV 也很常见，
    正好把这条路径打通。

输出:
    data/raw/force/<well>.csv        列: DEPTH,GR,RHOB,NPHI,DTC,SP,CALI,RDEP
    data/raw/force/lith_labels.csv   列: 井名,深度(m),岩性编码
"""

from __future__ import annotations

import os
import re

import pandas as pd

SRC = r"C:\油气藏多源数据表征\_cache\force2020\extracted\train.csv"
OUT = os.path.join("data", "raw", "force")

CURVES = {
    "GR": "GR",
    "RHOB": "RHOB",
    "NPHI": "NPHI",
    "DTC": "DTC",
    "SP": "SP",
    "CALI": "CALI",
    "RDEP": "RDEP",
}
LITH = "FORCE_2020_LITHOFACIES_LITHOLOGY"


def safe_name(well: str) -> str:
    """井名里有斜杠和空格，转成文件名安全的形式。"""
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", well).strip("_")


def main():
    os.makedirs(OUT, exist_ok=True)
    use = ["WELL", "DEPTH_MD", LITH, *CURVES.values()]
    print(f"读取 {SRC} ...")
    df = pd.read_csv(SRC, sep=";", usecols=use, low_memory=False)
    print(f"  {len(df):,} 行, {df['WELL'].nunique()} 口井")

    labels_all = []
    wells = sorted(df["WELL"].dropna().unique())
    for i, w in enumerate(wells, 1):
        sub = df[df["WELL"] == w].sort_values("DEPTH_MD").reset_index(drop=True)

        out = pd.DataFrame({"DEPTH": sub["DEPTH_MD"].to_numpy()})
        for std, src in CURVES.items():
            out[std] = sub[src].to_numpy()

        keep = out["DEPTH"].notna().to_numpy()
        out = out[keep].reset_index(drop=True)
        if len(out) < 500:
            continue
        out.to_csv(os.path.join(OUT, f"{safe_name(w)}.csv"), index=False)

        lb = pd.DataFrame(
            {
                "井名": safe_name(w),
                "深度(m)": out["DEPTH"].to_numpy(),
                "岩性编码": sub.loc[keep, LITH].to_numpy(),
            }
        )
        labels_all.append(lb.dropna())
        if i % 10 == 0 or i == len(wells):
            print(f"  已处理 {i}/{len(wells)}")

    lab = pd.concat(labels_all, ignore_index=True)
    lab["岩性编码"] = lab["岩性编码"].astype(int)
    lab.to_csv(os.path.join(OUT, "lith_labels.csv"), index=False)

    print(f"\n写出 {len(wells)} 口井的曲线文件 → {OUT}")
    print(f"标签 {len(lab):,} 行，{lab['岩性编码'].nunique()} 类")
    print("\n岩性分布:")
    vc = lab["岩性编码"].value_counts().sort_index()
    for c, n in vc.items():
        print(f"  {c:>7} {n:>8,} {n / len(lab):>6.1%}")


if __name__ == "__main__":
    main()
