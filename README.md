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
