"""配置加载与校验。

任何"换数据要改代码"的隐患，都应该在这里变成一条明确的报错，
而不是在后续流程里变成 KeyError 或静默的错误结果。
"""

from __future__ import annotations

import os
from typing import Any

import yaml

DEFAULT_CONFIG = os.path.join("configs", "data.yaml")


class ConfigError(Exception):
    """配置不合法。报错信息要能直接告诉用户改哪一行。"""


def load_config(path: str = DEFAULT_CONFIG) -> dict[str, Any]:
    if not os.path.exists(path):
        raise ConfigError(f"配置文件不存在: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ConfigError(f"配置文件格式错误，应为 YAML 字典: {path}")

    for key in ("window", "channels", "datasets", "target_step_m"):
        if key not in cfg:
            raise ConfigError(f"配置缺少必需字段: {key}")

    if not isinstance(cfg["channels"], dict) or not cfg["channels"]:
        raise ConfigError("channels 必须是非空字典，形如 {GR: [GR, GAM]}")
    for std, aliases in cfg["channels"].items():
        if not isinstance(aliases, list) or not aliases:
            raise ConfigError(f"channels.{std} 必须是非空列表（原始曲线名的别名）")

    if not isinstance(cfg["datasets"], dict) or not cfg["datasets"]:
        raise ConfigError("datasets 必须是非空字典")
    return cfg


def get_dataset(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    """取出某个数据集的配置，并把全局参数继承进去。"""
    if name not in cfg["datasets"]:
        available = ", ".join(cfg["datasets"].keys())
        raise ConfigError(f"未知数据集 '{name}'。可选: {available}")

    ds = dict(cfg["datasets"][name] or {})
    # 继承全局参数，数据集内可覆盖
    ds.setdefault("window", cfg["window"])
    ds.setdefault("target_step_m", cfg["target_step_m"])
    ds.setdefault("channels", cfg["channels"])
    ds.setdefault("allow_missing_channels", cfg.get("allow_missing_channels", True))
    ds.setdefault("min_channels", cfg.get("min_channels", 3))
    ds.setdefault("paths", {"raw": "data/raw", "processed": "data/processed"})

    # 井清单可以手写，也可以按目录自动扫描（井很多时用后者）
    if "auto_wells" in ds and ds["auto_wells"]:
        aw = ds["auto_wells"]
        pattern = aw.get("pattern", "*.csv")
        exclude = {str(x).lower() for x in aw.get("exclude", [])}
        raw_dir = ds.get("paths", {}).get("raw", "data/raw")
        if not os.path.isdir(raw_dir):
            raise ConfigError(f"datasets.{name}.auto_wells: 目录不存在 {raw_dir}")
        files = sorted(
            f
            for f in os.listdir(raw_dir)
            if f.lower().endswith(pattern.replace("*", "")) and f.lower() not in exclude
        )
        if not files:
            raise ConfigError(f"datasets.{name}.auto_wells: {raw_dir} 下没有匹配 {pattern} 的文件")
        key = "csv" if pattern.endswith(".csv") else "las"
        ds["wells"] = [
            {"name": os.path.splitext(f)[0], key: f, "depth_col": aw.get("depth_col", "DEPTH")}
            for f in files
        ]

    if "wells" not in ds or not ds["wells"]:
        raise ConfigError(f"datasets.{name} 缺少 wells 列表（或 auto_wells 配置）")
    for w in ds["wells"]:
        if "name" not in w or ("las" not in w and "csv" not in w):
            raise ConfigError(f"datasets.{name}.wells 每项都需要 name 与 las（或 csv）字段: {w}")

    label = ds.get("label", {"source": "none"})
    if label.get("source") not in ("none", "csv", "las_curve", None):
        raise ConfigError(
            f"datasets.{name}.label.source 只能是 none / csv / las_curve，"
            f"当前为 {label.get('source')!r}"
        )
    ds["label"] = label
    ds.setdefault("task", "regression")
    if ds["task"] not in ("regression", "classification"):
        raise ConfigError(
            f"datasets.{name}.task 只能是 regression / classification，当前 {ds['task']!r}"
        )
    return ds
