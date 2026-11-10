"""FORCE 2020 数据画像：井数、行数、岩性分布、曲线覆盖率、缺失情况。"""

import pandas as pd

CSV = r"C:\油气藏多源数据表征\_cache\force2020\extracted\train.csv"

USE = [
    "WELL",
    "DEPTH_MD",
    "GR",
    "RHOB",
    "NPHI",
    "DTC",
    "SP",
    "CALI",
    "RDEP",
    "FORCE_2020_LITHOFACIES_LITHOLOGY",
    "FORCE_2020_LITHOFACIES_CONFIDENCE",
]

print("读取中（267 MB）...")
df = pd.read_csv(CSV, sep=";", usecols=USE, low_memory=False)
print(f"行数 {len(df):,}，列数 {df.shape[1]}")

print(f"\n井数: {df['WELL'].nunique()}")
per = df.groupby("WELL").agg(
    n=("DEPTH_MD", "size"), top=("DEPTH_MD", "min"), bot=("DEPTH_MD", "max")
)
print(
    f"每井行数: 中位 {int(per['n'].median())}, "
    f"最少 {int(per['n'].min())}, 最多 {int(per['n'].max())}"
)
print(f"井段长度: 中位 {float((per['bot'] - per['top']).median()):.0f} m")

LITH = "FORCE_2020_LITHOFACIES_LITHOLOGY"
print(f"\n岩性编码分布（共 {df[LITH].nunique()} 类）:")
vc = df[LITH].value_counts().sort_index()
tot = len(df)
for code, cnt in vc.items():
    print(f"  {int(code):>7}  {cnt:>8,}  {cnt / tot:>6.1%}")

print("\n曲线覆盖率（非空比例）:")
for c in ["GR", "RHOB", "NPHI", "DTC", "SP", "CALI", "RDEP"]:
    print(f"  {c:<6} {df[c].notna().mean():>6.1%}")

print("\n四曲线齐全的样本:")
core4 = df[["GR", "RHOB", "NPHI", "DTC"]].notna().all(axis=1)
print(f"  GR+RHOB+NPHI+DTC 全非空: {core4.mean():.1%}（{int(core4.sum()):,} 行）")
print(f"  且岩性非空: {(core4 & df[LITH].notna()).mean():.1%}")

print("\n置信度:", df["FORCE_2020_LITHOFACIES_CONFIDENCE"].value_counts(dropna=False).to_dict())
print("\n各井岩性种类数（前 5 多 / 前 5 少）:")
wl = df.groupby("WELL")[LITH].nunique().sort_values()
print("  少:", wl.head(5).to_dict())
print("  多:", wl.tail(5).to_dict())
