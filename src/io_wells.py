"""数据适配层：把各种格式的原始文件读成统一结构。

下游（窗口化、训练、评测）只看到 Well / LabelSet，看不到 LAS 或 CSV。
换数据源时改的是 configs/data.yaml，不是这里。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import lasio
import numpy as np
import pandas as pd

FT_TO_M = 0.3048


@dataclass
class LabelSet:
    """岩心实测标签，独立于测井网格存在（真实数据里二者通常不对齐）。"""

    depth: np.ndarray
    value: np.ndarray
    lith: np.ndarray | None = None
    unit: str = "fraction"

    def __len__(self) -> int:
        return int(self.depth.size)


@dataclass
class Well:
    name: str
    depth: np.ndarray
    curves: dict[str, np.ndarray]
    labels: LabelSet | None = None
    source: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def n_points(self) -> int:
        return int(self.depth.size)

    @property
    def step(self) -> float:
        return float(np.median(np.diff(self.depth))) if self.depth.size > 1 else 0.0


def match_channels(las, alias_map: dict[str, list[str]]) -> tuple[dict[str, str], list[str]]:
    """按别名表把原始曲线名映射到标准名。

    匹配顺序：完全相同（忽略大小写） -> 前缀匹配。
    返回 ({标准名: 原始名}, [未匹配上的标准名])
    """
    have = {c.mnemonic.upper(): c.mnemonic for c in las.curves}
    picked: dict[str, str] = {}
    missing: list[str] = []

    for std, aliases in alias_map.items():
        hit = None
        for alias in aliases:
            a = alias.upper()
            if a in have:
                hit = have[a]
                break
        if hit is None:
            for alias in aliases:
                a = alias.upper()
                cand = next((v for k, v in have.items() if k.startswith(a)), None)
                if cand:
                    hit = cand
                    break
        if hit:
            picked[std] = hit
        else:
            missing.append(std)
    return picked, missing


def _load_logs_csv(path: str, alias_map: dict, depth_col: str = "DEPTH"):
    """从 CSV 读曲线（真实数据常以表格导出，而非 LAS）。

    返回 (depth, {标准名: ndarray}, 未匹配的标准名列表)。
    """
    df = pd.read_csv(path)
    cols = {str(c).strip().upper(): c for c in df.columns}
    if depth_col.strip().upper() not in cols:
        raise KeyError(
            f"{os.path.basename(path)} 缺少深度列 {depth_col!r}；现有列: {list(df.columns)[:12]}..."
        )
    depth = pd.to_numeric(df[cols[depth_col.strip().upper()]], errors="coerce").to_numpy(float)

    curves: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for std, aliases in alias_map.items():
        hit = None
        for alias in aliases:
            a = alias.strip().upper()
            if a in cols:
                hit = cols[a]
                break
        if hit is None:
            missing.append(std)
            continue
        curves[std] = pd.to_numeric(df[hit], errors="coerce").to_numpy(float)
    return depth, curves, missing


def _to_nan(arr: np.ndarray, null_value: float | None) -> np.ndarray:
    out = np.asarray(arr, dtype=float).copy()
    if null_value is not None and np.isfinite(null_value):
        out[np.isclose(out, null_value, atol=1e-6)] = np.nan
    out[out <= -900] = np.nan
    return out


def _resample(depth: np.ndarray, curves: dict[str, np.ndarray], target_step: float):
    """统一到 target_step。步长已一致则原样返回，避免无谓精度损失。"""
    if depth.size < 2:
        return depth, curves, False
    cur = float(np.median(np.diff(depth)))
    if abs(cur - target_step) <= target_step * 1e-3:
        return depth, curves, False

    new_depth = np.arange(depth[0], depth[-1] + target_step * 0.5, target_step)
    new_curves = {}
    for k, v in curves.items():
        ok = np.isfinite(v)
        if ok.sum() < 2:
            new_curves[k] = np.full(new_depth.size, np.nan)
        else:
            new_curves[k] = np.interp(new_depth, depth[ok], v[ok], left=np.nan, right=np.nan)
    return new_depth, new_curves, True


def _read_labels_csv(csv_path: str, well_name: str, label_cfg: dict) -> LabelSet | None:
    for col_key in ("well_col", "depth_col", "value_col"):
        if col_key not in label_cfg:
            raise KeyError(f"label 配置缺少 {col_key}（CSV 标签必需）")

    df = pd.read_csv(csv_path)
    need = [label_cfg["well_col"], label_cfg["depth_col"], label_cfg["value_col"]]
    for c in need:
        if c not in df.columns:
            raise KeyError(
                f"{os.path.basename(csv_path)} 中找不到列 {c!r}；"
                f"现有列: {list(df.columns)}。请改 configs/data.yaml 的 label 段"
            )

    sub = df[df[label_cfg["well_col"]].astype(str) == str(well_name)]
    if sub.empty:
        return None

    depth = pd.to_numeric(sub[label_cfg["depth_col"]], errors="coerce").to_numpy(float)
    value = pd.to_numeric(sub[label_cfg["value_col"]], errors="coerce").to_numpy(float)
    ok = np.isfinite(depth) & np.isfinite(value)

    lith = None
    lith_col = label_cfg.get("lith_col")
    if lith_col and lith_col in sub.columns:
        lith = sub[lith_col].astype(str).to_numpy()[ok]

    unit = label_cfg.get("unit", "fraction")
    if unit == "percent":
        value = value / 100.0
    return LabelSet(depth=depth[ok], value=value[ok], lith=lith, unit="fraction")


def load_well(well_cfg: dict, ds_cfg: dict) -> Well:
    """读取一口井，返回与原始格式无关的统一结构。"""
    name = well_cfg["name"]
    raw_dir = ds_cfg["paths"]["raw"]
    notes: list[str] = []

    src_name = well_cfg.get("las") or well_cfg.get("csv")
    if not src_name:
        raise KeyError(f"{name}: wells 项需要 las 或 csv 字段")
    src_path = os.path.join(raw_dir, src_name)
    if not os.path.exists(src_path):
        raise FileNotFoundError(
            f"找不到数据文件: {src_path}\n"
            f"  检查 configs/data.yaml 里数据集 paths.raw 与 wells 的文件名是否匹配"
        )

    is_csv = "csv" in well_cfg or well_cfg.get("fmt") == "csv"
    if is_csv:
        depth, curves_raw, missing = _load_logs_csv(
            src_path, ds_cfg["channels"], well_cfg.get("depth_col", "DEPTH")
        )
        picked = dict(curves_raw)
        null_value = None
        idx_unit = ""
        las = None
        notes.append("CSV 曲线输入")
    else:
        las = lasio.read(src_path)
        depth = np.asarray(las.index, dtype=float)
        picked, missing = match_channels(las, ds_cfg["channels"])
        curves_raw = {std: las[raw].data for std, raw in picked.items()}
        try:
            null_value = float(las.well.NULL.value)
        except Exception:
            null_value = None
        idx_unit = str(getattr(las.curves[0], "unit", "") or "").strip().lower()

    # 深度方向归一：真实数据里存在深度递减的井
    if depth.size > 1 and depth[0] > depth[-1]:
        depth = depth[::-1]
        flip = True
        notes.append("深度原为递减，已反转为递增")
    else:
        flip = False

    # 深度单位换算
    want = ds_cfg.get("units", {}).get("depth", "m").lower()
    if idx_unit in ("ft", "f", "feet") and want == "m":
        depth = depth * FT_TO_M
        notes.append("深度由 ft 换算为 m")

    if len(picked) < ds_cfg["min_channels"]:
        raise ValueError(
            f"{name}: 只匹配到 {len(picked)} 条通道（要求 ≥{ds_cfg['min_channels']}）。\n"
            f"  已匹配: {picked}\n"
            f"  未匹配: {missing}\n"
            f"  处理办法：在 configs/data.yaml 的 channels 对应项里补上该井的曲线名，不要改代码。"
        )
    if missing and not ds_cfg["allow_missing_channels"]:
        raise ValueError(f"{name}: 缺少通道 {missing}（allow_missing_channels=False）")
    if missing:
        notes.append(f"未匹配通道: {', '.join(missing)}")

    curves = {}
    for std, arr in curves_raw.items():
        curves[std] = _to_nan(arr, null_value)

    # 标签要按原始深度轴取（重采样会把稀疏的岩心列插值成假数据）
    depth_orig = depth.copy()

    depth, curves, resampled = _resample(depth, curves, float(ds_cfg["target_step_m"]))
    if resampled:
        notes.append(f"重采样至 {ds_cfg['target_step_m']} m")

    labels = None
    label_cfg = ds_cfg.get("label") or {"source": "none"}
    src = label_cfg.get("source")
    if src == "las_curve":
        curve = label_cfg.get("curve")
        if las is not None and curve and curve in {c.mnemonic for c in las.curves}:
            raw_lbl = _to_nan(las[curve].data, null_value)
            if flip:
                raw_lbl = raw_lbl[::-1]
            ok = np.isfinite(raw_lbl)
            if ok.sum() == 0:
                labels = None
                notes.append(f"标签曲线 {curve} 全为空")
            else:
                vals = raw_lbl[ok]
                if label_cfg.get("unit", "fraction") == "percent":
                    vals = vals / 100.0
                labels = LabelSet(depth=depth_orig[ok], value=vals, unit="fraction")
                notes.append(f"标签取自 LAS 曲线 {curve}，{int(ok.sum())} 个样品")
        else:
            notes.append(f"标签曲线 {curve} 不存在，跳过")
    elif src == "csv":
        csv_path = os.path.join(raw_dir, label_cfg["file"])
        if os.path.exists(csv_path):
            labels = _read_labels_csv(csv_path, name, label_cfg)
            if labels is None:
                notes.append(f"CSV 中无本井 {name} 的标签")
            else:
                notes.append(f"标签来自 {label_cfg['file']}，{len(labels)} 个样品")
        else:
            notes.append(f"标签文件不存在: {csv_path}")

    return Well(name=name, depth=depth, curves=curves, labels=labels, source=src_path, notes=notes)
