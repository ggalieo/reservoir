"""生成一个可执行的分析 notebook（读 LAS → 质控 → 出图 → 窗口化）。"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# 01 测井数据探索与窗口化\n\n"
    "配套《油气藏多源数据智能表征方法与系统研究》。\n\n"
    "**用途**：拿到一口新井的 LAS 后，先在这里看清它长什么样、缺什么、能切多少窗口，再进模型。\n\n"
    "**内核**：选 `Python (reservoir-repr)`。"))

cells.append(nbf.v4.new_code_cell(
    "import sys, warnings\n"
    "warnings.filterwarnings('ignore')\n"
    "import numpy as np, pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import torch, lasio\n"
    "\n"
    "plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']\n"
    "plt.rcParams['axes.unicode_minus'] = False\n"
    "%matplotlib inline\n"
    "\n"
    "print('python      ', sys.version.split()[0])\n"
    "print('numpy       ', np.__version__)\n"
    "print('pandas      ', pd.__version__)\n"
    "print('torch       ', torch.__version__)\n"
    "print('cuda 可用   ', torch.cuda.is_available())\n"
    "if torch.cuda.is_available():\n"
    "    print('显卡        ', torch.cuda.get_device_name(0))"))

cells.append(nbf.v4.new_markdown_cell(
    "## 1  读取 LAS\n\n"
    "先自动定位到工程根目录（notebook 默认在 `notebooks/` 下运行）。\n\n"
    "没有真实数据时先用合成井：`make_demo_las.py` 会生成 `data/raw/DEMO-01.las`。"))

cells.append(nbf.v4.new_code_cell(
    "import os\n"
    "from pathlib import Path\n"
    "\n"
    "ROOT = Path.cwd().resolve()\n"
    "while ROOT != ROOT.parent and not (ROOT / 'data').is_dir():\n"
    "    ROOT = ROOT.parent\n"
    "os.chdir(ROOT)\n"
    "print('工程根目录:', ROOT)"))

cells.append(nbf.v4.new_code_cell(
    "from pathlib import Path\n"
    "\n"
    "LAS_PATH = Path('data/raw/DEMO-01.las')\n"
    "assert LAS_PATH.exists(), '先运行 python make_demo_las.py 生成合成井'\n"
    "\n"
    "las = lasio.read(LAS_PATH)\n"
    "well = str(las.well.WELL.value)\n"
    "depth = np.asarray(las.index, dtype=float)\n"
    "step = float(np.median(np.diff(depth)))\n"
    "\n"
    "print(f'井名   : {well}')\n"
    "print(f'深度   : {depth[0]:.2f} – {depth[-1]:.2f} m，{depth.size} 点')\n"
    "print(f'采样   : {step:.3f} m')\n"
    "print(f'曲线   : {\", \".join(c.mnemonic for c in las.curves)}')"))

cells.append(nbf.v4.new_markdown_cell("## 2  质量概览\n\n缺失率、值域、尖峰数量。"))

cells.append(nbf.v4.new_code_cell(
    "curves = {}\n"
    "for c in las.curves:\n"
    "    arr = np.asarray(c.data, dtype=float)\n"
    "    arr = np.where(arr <= -900, np.nan, arr)\n"
    "    curves[c.mnemonic] = arr\n"
    "\n"
    "rows = []\n"
    "for k, a in curves.items():\n"
    "    if k == 'DEPT':\n"
    "        continue\n"
    "    v = a[np.isfinite(a)]\n"
    "    rows.append({\n"
    "        '通道': k,\n"
    "        '缺失率': f'{np.isnan(a).mean():.1%}',\n"
    "        '最小': round(float(v.min()), 3) if v.size else None,\n"
    "        '最大': round(float(v.max()), 3) if v.size else None,\n"
    "        '点数': int(np.isfinite(a).sum()),\n"
    "    })\n"
    "pd.DataFrame(rows)"))

cells.append(nbf.v4.new_markdown_cell("## 3  四道曲线图\n\n缺失段会自动涂灰。"))

cells.append(nbf.v4.new_code_cell(
    "show = [k for k in ('GR', 'RT', 'DEN', 'AC') if k in curves]\n"
    "fig, axes = plt.subplots(1, len(show), figsize=(3 * len(show), 7), sharey=True)\n"
    "for ax, k in zip(axes, show):\n"
    "    ax.plot(curves[k], depth, lw=0.8, color='#1f4e79')\n"
    "    ax.set_title(k)\n"
    "    ax.grid(alpha=0.3)\n"
    "axes[0].invert_yaxis()\n"
    "axes[0].set_ylabel('深度 (m)')\n"
    "plt.tight_layout()\n"
    "plt.show()"))

cells.append(nbf.v4.new_markdown_cell(
    "## 4  窗口化\n\n"
    "按 2 m / 50% 重叠切窗。留意每窗的深度点数 —— 2 m ÷ 0.125 m = **16 点**，\n"
    "这正是 Transformer 在这套参数下发挥不出长程建模能力的原因，正式模型要用更长的上下文窗口。"))

cells.append(nbf.v4.new_code_cell(
    "WINDOW_M, OVERLAP = 2.0, 0.5\n"
    "names = [k for k in curves if k not in ('DEPT', 'CPOR')]\n"
    "L = int(round(WINDOW_M / step))\n"
    "stride = max(1, int(round(L * (1 - OVERLAP))))\n"
    "arr = np.stack([curves[k] for k in names])\n"
    "n_win = max(0, (depth.size - L) // stride + 1)\n"
    "\n"
    "win = np.empty((n_win, len(names), L), dtype=np.float32)\n"
    "presence = np.ones((n_win, len(names)), dtype=np.int8)\n"
    "for i in range(n_win):\n"
    "    s = i * stride\n"
    "    block = arr[:, s:s + L]\n"
    "    win[i] = block\n"
    "    presence[i] = (np.isfinite(block).sum(axis=1) > 0).astype(np.int8)\n"
    "\n"
    "print(f'窗口数 {n_win}，形状 {win.shape}（窗口 × 通道 × 深度点）')\n"
    "print(f'每窗深度点数 {L}，步长 {stride}')\n"
    "print(f'窗口内缺失采样占比 {np.isnan(win).mean():.1%}')"))

cells.append(nbf.v4.new_markdown_cell("## 5  存盘\n\n产出可直接被训练脚本读取。"))

cells.append(nbf.v4.new_code_cell(
    "out = Path('data/processed') / f'{well}_win.npz'\n"
    "out.parent.mkdir(parents=True, exist_ok=True)\n"
    "np.savez_compressed(out, win=win, presence=presence, channels=np.array(names))\n"
    "print('saved ->', out)\n"
    "\n"
    "chk = np.load(out)\n"
    "print('读取校验:', chk['win'].shape, chk['channels'].tolist())"))

cells.append(nbf.v4.new_markdown_cell(
    "---\n\n"
    "### 换成真实井\n\n"
    "把 `LAS_PATH` 改成 `data/raw/W-01.las` 之类的真实文件重跑即可；\n"
    "若曲线命名与默认不一致（如 `RHOB` 而非 `DEN`），通道匹配逻辑见 `smoke_las.py` 的 `match_channels`。"))

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {
    "display_name": "Python (reservoir-repr)",
    "language": "python",
    "name": "reservoir-repr",
}
nb["metadata"]["language_info"] = {"name": "python", "version": "3.12.3"}

path = r"C:\油气藏多源数据表征\reservoir-repr\notebooks\01_explore_las.ipynb"
nbf.write(nb, path)
print("wrote", path, "cells:", len(cells))
