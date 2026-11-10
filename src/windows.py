"""窗口化与标签归属。

设计约定（写进论文的"数据组织方式"）：
  - 窗口按固定深度网格切分，步长 = 长度 × (1 - 重叠)
  - 窗口内所有岩心点取均值作为标签；无岩心点则 y = NaN（不是 0）
  - 记录每个窗口的岩心点数与岩性一致性，供后续筛选/加权
"""

from __future__ import annotations

import numpy as np

from .io_wells import LabelSet, Well


def make_windows(well: Well, channels: list[str], length_m: float, overlap: float):
    """把一口井切成窗口。

    返回:
        win      (N, C, L)  曲线窗口，缺失通道整列 NaN
        presence (N, C)     该通道在此窗口是否有观测
        starts   (N,)       窗口起始深度
        centers  (N,)       窗口中心深度
    """
    step = well.step
    if step <= 0:
        raise ValueError(f"{well.name}: 深度轴异常（step={step}）")

    L = int(round(length_m / step))
    if L < 2:
        raise ValueError(
            f"{well.name}: 窗口长度 {length_m} m 在 {step} m 步长下只有 {L} 点，太短。"
            f"检查 configs/data.yaml 的 window.length_m"
        )
    stride = max(1, int(round(L * (1.0 - overlap))))

    n = max(0, (well.n_points - L) // stride + 1)
    if n == 0:
        raise ValueError(
            f"{well.name}: 无法切窗（{well.n_points} 点 < 窗口 {L} 点）。检查数据长度或窗口长度"
        )

    C = len(channels)
    arr = np.full((C, well.n_points), np.nan, dtype=float)
    for i, ch in enumerate(channels):
        if ch in well.curves:
            arr[i] = well.curves[ch]

    win = np.empty((n, C, L), dtype=np.float32)
    presence = np.zeros((n, C), dtype=np.int8)
    starts = np.empty(n, dtype=float)

    for i in range(n):
        s = i * stride
        block = arr[:, s : s + L]
        win[i] = block
        presence[i] = (np.isfinite(block).sum(axis=1) > 0).astype(np.int8)
        starts[i] = well.depth[s]

    centers = starts + (L - 1) * step / 2.0
    return win, presence, starts, centers


def attach_labels(
    starts: np.ndarray,
    length_m: float,
    labels: LabelSet | None,
    lith_code: dict[str, int] | None = None,
):
    """把岩心标签按深度归属到窗口（窗内取均值）。

    返回 y / n_label / lith / lith_consistent，均为 (N,)
    """
    n = starts.size
    y = np.full(n, np.nan, dtype=np.float32)
    n_label = np.zeros(n, dtype=np.int16)
    lith = np.full(n, -1, dtype=np.int16)
    lith_consistent = np.zeros(n, dtype=np.int8)

    if labels is None or len(labels) == 0:
        return y, n_label, lith, lith_consistent

    d = np.asarray(labels.depth, dtype=float)
    v = np.asarray(labels.value, dtype=float)
    lith_names = labels.lith

    for i, s in enumerate(starts):
        m = (d >= s) & (d < s + length_m)
        k = int(m.sum())
        if k == 0:
            continue
        vals = v[m]
        ok = np.isfinite(vals)
        if ok.sum() == 0:
            continue
        y[i] = float(np.mean(vals[ok]))
        n_label[i] = int(ok.sum())

        if lith_names is not None and lith_code:
            names = lith_names[m][ok]
            uniq = np.unique(names)
            lith_consistent[i] = int(uniq.size == 1)
            # 多数票
            vals_u, counts = np.unique(names, return_counts=True)
            lith[i] = lith_code.get(str(vals_u[np.argmax(counts)]), -1)

    return y, n_label, lith, lith_consistent


def summarize(name: str, win, presence, y, n_label, lith_consistent) -> dict:
    """生成一条可打印/可入库的统计。"""
    n = win.shape[0]
    labeled = int(np.isfinite(y).sum())
    yv = y[np.isfinite(y)]
    return {
        "well": name,
        "n_windows": n,
        "n_labeled": labeled,
        "labeled_ratio": labeled / n if n else 0.0,
        "n_channels": win.shape[1],
        "window_len_pts": win.shape[2],
        "channel_coverage": (presence.mean(axis=0) if n else np.zeros(presence.shape[1])).tolist(),
        "y_min": float(yv.min()) if yv.size else None,
        "y_max": float(yv.max()) if yv.size else None,
        "y_mean": float(yv.mean()) if yv.size else None,
        "y_std": float(yv.std()) if yv.size else None,
        "labels_per_window_mean": float(n_label.mean()) if n else 0.0,
        "lith_consistent_ratio": (
            float(lith_consistent[np.isfinite(y)].mean()) if labeled else None
        ),
    }
