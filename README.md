# 油气藏多源数据智能表征 — 实验工程

配套本科毕设《油气藏多源数据智能表征方法与系统研究》。

## 环境现状（已装好）

| 项 | 值 |
|---|---|
| Python | 3.12.3（venv，见 `.venv/`） |
| PyTorch | 2.6.0+cu124 ✅ CUDA 可用 |
| GPU | RTX 4060 Laptop，8 GB，算力验证通过 |
| 调用方式 | `.venv\Scripts\python.exe`，**不要用 Activate.ps1** |

> 本机 PowerShell 执行策略禁止 `.ps1`，所以一律用绝对路径调用解释器。

一次验证全部关键项：

```powershell
.venv\Scripts\python.exe -c "import torch,numpy,lasio,welly,segyio,sklearn; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

期望：`2.6.0+cu124 True NVIDIA GeForce RTX 4060 Laptop GPU`

## 装环境踩过的两个坑（重装时会再遇到）

**① `download.pytorch.org` 下不动。** 实测单连接 0.85 MB/s 且经常直接卡死。
解决：走阿里云镜像的分块并发下载，9 MB/s 左右，4.6 分钟拿完 2.5 GB。

```powershell
# 分块并发下载（脚本见 git 历史中的 fetch_wheel2.py 思路：32 MB 一块、8 并发）
# 然后本地安装：
.venv\Scripts\python.exe -m pip install .\torch-2.6.0+cu124-cp312-cp312-win_amd64.whl
```

注意：PyPI 上的 `torch` 在 Windows 是 **CPU 版**，`pip install torch` 装不到 CUDA。

**② `welly` 导入报 `No module named 'pkg_resources'`。**
torch 安装会顺带把 setuptools 升到 84，而 setuptools 84 移除了 `pkg_resources`，welly 依赖它。

```powershell
.venv\Scripts\python.exe -m pip install "setuptools<81"
```

## 一键恢复环境

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

（`requirements.txt` 里的 torch 需按上面①的方式单独装）

## 数据链路冒烟测试

没有真实数据时，先跑合成井验证整条链路：

```powershell
.venv\Scripts\python.exe make_demo_las.py
.venv\Scripts\python.exe smoke_las.py data\raw\DEMO-01.las
```

产出：
- `reports/smoke_DEMO-01.png` —— 四道曲线 + 岩心孔隙度，缺失段涂灰
- `data/processed/DEMO-01_win.npz` —— 窗口数组 `(N, C, L)` + 通道存在性标记

拿到真实 LAS 后换路径即可：

```powershell
.venv\Scripts\python.exe smoke_las.py data\raw\W-01.las --window 2.0 --overlap 0.5
```

## Jupyter

已装 JupyterLab 4.6.6，并把本 venv 注册为内核 `Python (reservoir-repr)`。

启动：

```powershell
cd C:\油气藏多源数据表征\reservoir-repr
.venv\Scripts\python.exe -m jupyter lab --notebook-dir=.
```

浏览器会自动打开。**新建 notebook 时务必选 `Python (reservoir-repr)` 内核**，
选成默认的 `python3` 会找不到 torch / lasio。

已有 notebook：`notebooks/01_explore_las.ipynb`（读 LAS → 质控 → 四道曲线图 → 窗口化 → 存盘）。
其中已内置"自动定位工程根目录"的逻辑，所以在 `notebooks/` 子目录下打开也不会找不到 `data/`。

无界面执行（CI / 批量验证）：

```powershell
.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace notebooks\01_explore_las.ipynb
```

## 目录

```
reservoir-repr/
├── .venv/                    # 虚拟环境（不入库）
├── configs/                  # Hydra 配置（待建）
├── data/
│   ├── raw/                  # 原始 LAS / SEG-Y（不入库）
│   └── processed/            # 窗口化产物
├── notebooks/                # 探索性分析
├── reports/                  # 图件
├── runs/                     # 运行记录（配置快照 + 日志）
├── src/                      # 正式代码（待建）
├── make_demo_las.py          # 合成井生成器
├── smoke_las.py              # 冒烟测试：读入→质控→出图→窗口化
├── fetch_wheel2.py           # 分块并发下载 torch 轮子
└── requirements.txt
```

## 冒烟测试验证了什么

| 检查项 | 通过标志 |
|---|---|
| LAS 读取 | 打印曲线数与深度范围 |
| 通道自动匹配 | 按别名匹配（GR/RHOB/DT 等不同命名均可） |
| 质控 | 缺失率、稳健 z 尖峰数、值域逐通道给出 |
| 窗口化 | 输出 `(N, C, L)` 与 presence 标记 |
| 出图 | 缺失段与异常段肉眼可辨 |

合成井实测：7/7 通道识别，60 m 切出 59 个 `(N,7,16)` 窗口，质控捕获注入的 15 处井径异常与 5 处电阻率尖峰。

## 下一步

1. 从 FORCE 2020 取真实井，替换 `data/raw/` 下的合成井
2. 把 `smoke_las.py` 里的函数拆进 `src/pipeline.py`
3. 建 `configs/` 与 `src/train.py`，跑通第一个基线（RandomForest）
