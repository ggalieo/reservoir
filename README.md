# 油气藏多源数据智能表征 — 实验工程

配套本科毕设《油气藏多源数据智能表征方法与系统研究》。

当前使用 **FORCE 2020 公开真实数据**（98 口井，挪威北海），任务为岩性分类。
合成/模拟数据已从工程中移除，git 历史中仍可追溯。

## 环境

| 项 | 值 |
|---|---|
| Python | 3.12.3（venv，见 `.venv/`） |
| PyTorch | 2.6.0+cu124，CUDA 可用 |
| GPU | RTX 4060 Laptop 8 GB |
| 调用方式 | `.venv\Scripts\python.exe`，**不要用 Activate.ps1** |

验证：

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 数据：FORCE 2020

原始数据不在仓库中（`data/` 已忽略）。获取与转换：

```powershell
# 1. 下载（raw.githubusercontent 被限速，用镜像）
curl.exe -L --ssl-no-revoke -o train.zip `
  "https://gh-proxy.com/https://raw.githubusercontent.com/bolgebrygg/Force-2020-Machine-Learning-competition/master/lithology_competition/data/train.zip"

# 2. 解压后把 train.csv 放到 _cache/force2020/extracted/
# 3. 转成每井一个 CSV + 岩性标签表
.venv\Scripts\python.exe tools\convert_force.py
```

产物：`data/raw/force/` 下 98 个井 CSV + `lith_labels.csv`。

数据规模：1,170,511 行 / 98 井 / 12 类岩性。

曲线覆盖（真实数据的"脏"）：

| 曲线 | 覆盖率 |
|---|---|
| GR | 100.0% |
| RDEP | 99.1% |
| DTC | 93.1% |
| CALI | 92.5% |
| RHOB | 86.2% |
| **NPHI** | **65.4%** |

四曲线齐全仅 **61.6%** —— 天然具备模态缺失场景，做鲁棒性实验无需人工注入。

## 构建数据集

```powershell
.venv\Scripts\python.exe build_dataset.py --dataset force2020
```

**换数据只改 `configs/data.yaml`，代码一行不动。** 数据源差异全部由配置吸收：

| 差异点 | 配置项 |
|---|---|
| 曲线叫 `CALI` 还是 `CAL`、`RDEP` 还是 `RT` | `channels` 别名表 |
| 曲线在 LAS 里还是在 CSV 里 | wells 项用 `las` 或 `csv` 字段 |
| 标签在 CSV 里还是在 LAS 某列里 | `label.source: csv \| las_curve \| none` |
| 监督任务类型 | `task: regression \| classification` |
| 井很多（98 口） | `auto_wells` 按目录扫描 |
| 深度/孔隙度单位 | `units.depth`、`label.unit` |

输出 `data/processed/<dataset>_labeled_windows.npz`：

| 字段 | 形状 | 含义 |
|---|---|---|
| `win` | (N, C, L) | 曲线窗口 |
| `presence` | (N, C) | 该通道在此窗口是否有观测 |
| `y` | (N,) | 回归：孔隙度；分类：类别码。**无标签为 NaN，不是 0** |
| `n_label` | (N,) | 窗口内标签点数 |
| `depth_center` / `well` | (N,) | 窗口中心深度 / 井名 |
| `channels` | (C,) | 通道顺序（全数据集统一） |
| `task` / `class_ids` | (1,) / (K,) | 任务类型 / 分类类别码 |

## 基线结果

```powershell
.venv\Scripts\python.exe -m src.baseline_rf --dataset force2020 --folds 5
```

RF + 42 维窗口统计量，按井分组 5 折：

| 指标 | 值 |
|---|---|
| Accuracy | 0.697 |
| **Balanced accuracy** | **0.369** |
| **F1-macro** | **0.391** |

瓶颈在稀有岩性：大类召回 0.72–0.86，稀有类召回 0.00–0.19（如 65030 = 0.154）。
混淆矩阵见 `reports/force2020_baseline_cm.png`。

**这是本课题的靶子**：表征学习的价值不在提升大类精度，而在让小样本类可分。

## 目录

```
reservoir-repr/
├── configs/data.yaml        # 数据集定义（换数据只改这里）
├── data/raw/force/          # FORCE 井 CSV + 标签（不入库）
├── data/processed/          # 窗口化产物（不入库）
├── reports/                 # 基线结果与图件（入库）
├── src/
│   ├── config.py            # 配置加载与校验
│   ├── io_wells.py          # LAS / CSV 适配层
│   ├── windows.py           # 切窗与标签归属
│   └── baseline_rf.py       # RF 基线（回归/分类）
├── tools/convert_force.py   # FORCE 数据转换
├── build_dataset.py         # 数据集构建入口
├── profile_force.py         # 数据画像
├── demo_torch.py            # 环境自检（随机数据，非研究用）
├── smoke_las.py             # LAS 格式检查工具（需显式路径）
└── PROGRESS.md              # 进度表
```

## 下一步

见 `PROGRESS.md` 的待办：掩码自监督预训练 → 鲁棒性实验 → 井震跨模态 → 系统。
