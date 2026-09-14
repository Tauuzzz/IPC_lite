# IPC Lite —— 教学版 IPC (Incremental Potential Contact)

一个用最小代码量演示 **IPC（增量势能接触）** 核心算法的教学项目。
用 Python + [NVIDIA Warp](https://github.com/NVIDIA/warp) 实现，全部注释为中文，
代码结构与算法伪代码一一对应，适合已经读过 IPC 论文、想看"论文公式如何落地成代码"的读者。

> **这不是生产代码。** 为了教学清晰，刻意牺牲了性能与工程健壮性（见下文"教学简化"）。

## 这个 demo 做什么

两块四面体网格立方体：下块底面固定（Dirichlet 边界），上块带横向偏移做自由落体，
砸到下块时 barrier 接触能激活，摩擦能抵抗横向滑动。每个时间步用 Newton 法
求解增量势能极小化，梯度由 Warp tape 自动微分得到。

## 快速开始

```bash
pip install -r requirements.txt
python main.py
```

有 NVIDIA GPU 会自动用 CUDA，没有则回退到 CPU（教学网格很小，CPU 也能跑）。

两个更小的单元示例：

```bash
python example_strain_energy.py      # 应变能：单个四面体的 F / 能量
python example_energy_gradient.py    # barrier + 摩擦能：两个四面体的接触
```

两个**可交互 notebook**（公式逐式拆解 + 滑块调参，需要 `pip install ipywidgets jupyter`）：

- `notebooks/barrier.ipynb` — Barrier 接触能：公式推导、d² 变量替换的原因、
  κ/d̂ 取舍的交互滑块、d→0 钳位保险丝、固定候选集的代价
- `notebooks/friction.ipynb` — 摩擦能：f0 光滑化（含 C1 接续条件验证）、
  滑动位移算子 Γ、λ 滞后冻结的取舍、交互滑块看 μ/λ 如何控制摩擦

notebook 里的公式全部用 numpy 复算，并和 warp kernel 做了数值一致性交叉验证。

## 代码地图（建议阅读顺序）

| 文件 | 内容 | 对应论文概念 |
|---|---|---|
| `geometry.py` | 网格生成、表面提取、PT/EE 候选对、点-面/边-边距离 | 距离计算与候选集 |
| `strain_energy.py` | 稳定 Neo-Hookean 应变能 kernel | 弹性势能 ψ(F) |
| `barrier_energy.py` | barrier 能量及其法向力 | E_b(d) = −κ(d−d̂)²·ln(d/d̂) |
| `friction_energy.py` | 滞后(lagged)摩擦数据与摩擦耗散能 | 切向摩擦势能 D(Δx) |
| `mass.py` | lumped 质量、动能（惯性势能）项 | 增量势能中的惯性项 |
| `energy.py` | 能量归约与合并（装配总能量） | E(x) = K + ψ + B + D |
| `newton_solver.py` | Newton 方向、FD Hessian、CCD、Armijo 线搜索 | 隐式时间积分求解器 |
| `main.py` | 主流程：初始化 + 时间步循环（带伪代码注释） | 算法总装 |
| `INDEX.md` | **变量对照表**：每个变量在哪定义、什么形状、什么含义 | — |
| `notebooks/barrier.ipynb` | barrier 公式交互拆解（滑块调 κ/d̂） | — |
| `notebooks/friction.ipynb` | 摩擦公式交互拆解（滑块调 μ/λ） | — |

`main.py` 顶部的 docstring 就是用中文重写的 IPC 算法伪代码，
代码中每个关键位置都标注了对应的伪代码编号（step 级 1~5、迭代内 1~11）。

## 核心算法流程（main.py 顶部伪代码的浓缩版）

```
每步迭代最小化增量势能:
    E(x) = ‖x−x̂‖²_M/2Δt²   (惯性)
         + ψ(F)              (弹性)
         + Σ B(d(x), d̂)      (barrier 接触)
         + Σ μ·λ·f0(‖ξ‖)     (摩擦, λ/ξ 用上一步滞后位置冻结)

Newton 迭代:  H·p = −g → CCD 限制步长防穿透 → Armijo 线搜索 → 更新 x
```

## Barrier 与摩擦：公式、取舍与近似

> 公式推导和交互图表见 [notebooks/barrier.ipynb](notebooks/barrier.ipynb) 和
> [notebooks/friction.ipynb](notebooks/friction.ipynb)，本节是文字版速览。

### Barrier 接触能：用"对数墙"代替"不可穿透约束"

接触在数学上是一个不等式约束 `d(x) ≥ 0`。精确处理要用约束求解器（LCP 等），
IPC 的取舍是**把约束变成能量**：距离越近能量越高，物体被"软墙"顶住。

论文原式（以距离 d 为变量）：

```
b(d) = −κ·(d − d̂)²·ln(d / d̂),   d < d̂
b(d) = 0,                        d ≥ d̂
```

对应的代码在 `barrier_energy.py`。本实现的取舍：

| 论文/真实 IPC | 本实现 | 为什么可以 |
|---|---|---|
| 对每对几何体精确算最近点距离 | PT/EE 距离 kernel，退化情形（三角形压扁、边共线）逐一特判 | 教学网格简单，退化处理写成显式分支更易读（`geometry.py`） |
| d → 0 时能量发散到 +∞，天然不穿透 | `d² ≤ 1e-12` 时钳到 `1e-12`（`barrier_energy.py:20`） | 纯保险丝：正常解里 d 停在 d̂ 量级；真穿透交给 CCD 拦截 |
| 每 step 用 broad-phase 重建候选集 | 初始化一次枚举全部 PT/EE 对，之后固定 | 网格小，固定候选集代码量减半；代价见"教学简化" |
| κ 自适应（barrier 与弹性能量量纲匹配） | 固定 `kappa = 1e5` | 固定值让"调 κ 看穿透/抖动"成为练习题 |

**注意 barrier 的变量是 d² 还是 d**：本实现与 IPC toolkit 一致，把公式写在
`u = d²` 上（`b(u) = −κ(u − d̂²)²·ln(u/d̂²)`），这样省掉每次开根号，
自动微分也不用处理 `sqrt` 在 0 点的无穷导数。

### 摩擦能：把"不滑动"变成"滑得越远越贵"

库仑摩擦是"切向力 ≤ μ·λ"的不等式，同样不方便放进能量极小化框架。
IPC 的做法是给切向滑动位移 y 定义一个耗散势能：

```
D(y) = μ·λ·f0(‖y‖)
f0(s) = s,                            s ≥ y_eps   （纯滑动区）
f0(s) = s²/y_eps − s³/(3·y_eps²) + y_eps/3,   s < y_eps   （静摩擦区）
```

四个关键近似（代码在 `friction_energy.py`）：

1. **λ 用滞后构型冻结（lagged/friction anchoring）**。
   法向力 λ = N(d) 本应是当前构型 x 的函数，那样目标函数会强烈非线性、Newton 难收敛。
   本实现在每个时间步开头，在上一步解 xⁿ 上把 λ、接触点参数（β/α）、切平面基
   （tangent0/1）全部算好冻结，整个 Newton 期间只读（`main.py` step 级 3）。
   代价：摩擦力对当前步的法向变化"晚一步"响应——这是 IPC 论文的标准做法。
2. **|y| 的 C1 光滑化**。原始 |y| 在 0 处有尖角（不可导），Newton 法需要至少一阶导连续。
   三次多项式段把尖角抹平，两段在 `y_eps` 处函数值和导数都相等（f0(y_eps)=y_eps,
   f0′(y_eps)=1），所以"静摩擦 → 滑动"过渡是平滑的。`y_eps = DT·eps_v` 把速度阈值
   换算成位移阈值。
3. **切平面投影**。相对位移只取切向分量（沿接触面），法向分量归 barrier 管，
   两套能量各司其职、互不重复计费。
4. **滑动位移算子 Γ**。接触点不一定在顶点上（点可能落在三角形内部/边上），
   相对位移用重心坐标（β 或 α）把四个顶点的位移混合出来——
   这就是 lagged 数据里存 β/α 的原因。

### 一图流总结

（曲线图见两个 notebook，已含 warp kernel 一致性验证）

| 现象 | 曲线表现 |
|---|---|
| 接触是"软墙"不是硬约束 | b(d) 在 d̂ 处平滑接入 0，d→0 时 log 陡增 |
| 力在激活瞬间连续、随后增大 | N(d) 从 d̂ 处 0 开始增长，越近越大 |
| 钳位保险丝的位置 | d < 1e-6 处曲线被截断（正常解到不了） |
| 静摩擦区 | f0 在 |y| < y_eps 内是抛物线，之外是直线 |
| 压得越紧越难滑 | λ 越大 D(y) 越陡 |


## 教学简化（与真实 IPC 的差距）

读代码前先明确这些刻意简化，避免被误导：

1. **候选集固定**：初始化时一次性枚举全部 PT/EE 对，之后不重建。
   真实 IPC 每步跑 broad-phase + 空间哈希，只保留近距离对。
2. **FD Hessian**：梯度用自动微分，Hessian 用中心差分（每自由度 2 次梯度求值），
   稠密 `np.linalg.solve`。真实 IPC 用解析 Hessian + 稀疏求解。
   只适合几百顶点的小网格。
3. **CCD 简化**：对步长 α 做二分采样做"体积>0 且不穿透"可行性检查，
   而不是精确求每对几何体的穿越时刻（保守但正确）。
4. **无阻尼/无 self-collision 专项处理**：barrier 只对候选对生效，
   不含 IPC 论文的连续碰撞全流程。
5. **float32 求梯度**：Warp 自动微分在 float32 上跑，Python 侧状态用 float64
   账本。这是"两边系统"（numpy 账本 + warp 计算数组）的根本原因。

## 前置知识

- 增量势能方法 / 隐式欧拉时间积分
- Barrier 能量与光滑接触力
- 一点点优化：Newton 法、线搜索、投影牛顿

## 参考

- [Li et al., 2020, **Incremental Potential Contact**: Intersection- and Inversion-free Large-Deformation Dynamics](https://dl.acm.org/doi/10.1145/3386569.3392425)（主论文）
- [Li et al., 2021, Codimensional Incremental Potential Contact (IPL)](https://ipc-sim.github.io/IPC/)（官方实现与文档）
- [Ferguson et al., 2021, Stable Neo-Hookean Flesh Simulation](https://graphics.cs.utah.edu/research/projects/flesh/)（本文使用的弹性模型）

## License

MIT，详见 [LICENSE](LICENSE)。
