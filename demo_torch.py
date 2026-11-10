"""PyTorch 冒烟演示：用 5 条测井曲线预测孔隙度，全程在 GPU 上跑。

目的不是精度，而是让你亲眼看到：
    1. 张量确实在 cuda 上
    2. 反向传播、优化器、损失下降都正常
    3. GPU 相对 CPU 快多少

用法:
    .venv\\Scripts\\python.exe demo_torch.py
"""

import time

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("=" * 58)
print(f"PyTorch {torch.__version__}")
print(f"设备     : {DEV}")
if DEV == "cuda":
    p = torch.cuda.get_device_properties(0)
    print(f"显卡     : {p.name}")
    print(f"显存     : {p.total_memory / 1024**3:.1f} GB")
    print(f"算力     : sm_{p.major}{p.minor}")
print("=" * 58)

# ---- 造一批"像测井"的数据：GR/RT/DEN/CNU/AC -> 孔隙度 ----
N = 20_000
gr = np.random.normal(60, 20, N)
rt = np.random.lognormal(2.0, 0.6, N)
den = np.random.normal(2.4, 0.08, N)
cnu = np.random.normal(0.20, 0.05, N)
ac = np.random.normal(75, 8, N)

# 真实的"物理关系"：密度与中子对孔隙度最敏感
poro = 0.42 * (2.65 - den) / 1.65 + 0.35 * cnu + 0.0004 * (ac - 60) + np.random.normal(0, 0.01, N)
poro = np.clip(poro, 0.02, 0.38)

X = np.stack([gr, rt, den, cnu, ac], axis=1).astype("float32")
y = poro.astype("float32").reshape(-1, 1)

# 标准化（真实项目里必须做，否则训练不稳）
MU = X.mean(0).astype("float32")
SD = X.std(0).astype("float32")
X = (X - MU) / (SD + 1e-8)

split = int(N * 0.8)
Xtr = torch.from_numpy(X[:split]).to(DEV)
ytr = torch.from_numpy(y[:split]).to(DEV)
Xte = torch.from_numpy(X[split:]).to(DEV)
yte = torch.from_numpy(y[split:]).to(DEV)
print(f"样本     : 训练 {split} / 测试 {N - split}")
print(f"输入张量 : {tuple(Xtr.shape)} on {Xtr.device}")

model = nn.Sequential(
    nn.Linear(5, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
).to(DEV)
print(f"模型     : {sum(p.numel() for p in model.parameters())} 参数")

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
lossf = nn.MSELoss()

t0 = time.time()
for epoch in range(1, 401):
    opt.zero_grad()
    pred = model(Xtr)
    loss = lossf(pred, ytr)
    loss.backward()
    opt.step()
    if epoch % 100 == 0:
        print(f"  epoch {epoch:>3}  loss {loss.item():.6f}")
if DEV == "cuda":
    torch.cuda.synchronize()
dt = time.time() - t0

with torch.no_grad():
    pt = model(Xte)
    rmse = float(torch.sqrt(lossf(pt, yte)))
    ss_res = float(((yte - pt) ** 2).sum())
    ss_tot = float(((yte - yte.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot

print("-" * 58)
print(f"训练耗时 : {dt:.2f} s（400 epoch，{DEV}）")
print(f"测试 RMSE: {rmse:.4f}")
print(f"测试 R2   : {r2:.4f}")
print("-" * 58)

# ---- 演示推理：随便给一口"新井"的几个深度点 ----
new = torch.tensor(
    [[55.0, 8.0, 2.35, 0.22, 80.0], [95.0, 3.0, 2.55, 0.10, 62.0]],
    dtype=torch.float32,
    device=DEV,
)
new = (new - torch.tensor(MU, device=DEV)) / (torch.tensor(SD, device=DEV) + 1e-8)
with torch.no_grad():
    print("推理示例 :", [f"{v:.3f}" for v in model(new).squeeze().tolist()])
print("\n[OK] PyTorch 可以正常使用了")
