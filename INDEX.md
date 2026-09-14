# IPC Lite 变量对照表 (INDEX)

学习版 IPC 的全部变量索引：代码里出现一个名字，来这里查它的含义、形状和位置。

## 命名规范

- `_np` 后缀 = CPU / numpy 侧（float64，Python 账本，**无梯度**）
- `_wp` 后缀 = GPU / warp 侧（float32 数组，kernel 输入，支持自动微分）
- Warp kernel 只碰 `_wp`，状态只存 `_np`，两边靠 `wp.copy` 交接
- 传统数学记号与学习笔记一致：`DT, d_tilde, kappa, mu, lamda, PT_pair, EE_pair ...`

---

## 1. 顶点状态（位置 / 速度 / 质量）

| 变量 | 含义 | 类型 / 形状 | 位置 |
|---|---|---|---|
| `positions_np` | 当前位置（状态本体，t=0 起从 rest 开始演进） | ndarray (n,3) float64 | main.py 循环 |
| `positions_wp` | 当前位置的 GPU 形态，写入猜测位置后自动求梯度 | wp.array vec3 (n,) requires_grad | main.py 构造，newton_solver 持有 |
| `positions_lagged_np` | 滞后位置 x^n 的 numpy 形态（step 开头的解快照，速度更新基准） | ndarray (n,3) float64 | main.py step 级 1 |
| `positions_lagged_wp` | 滞后位置 x^n 的 GPU 形态（摩擦锚点，Newton 期间冻结） | wp.array vec3 (n,) | main.py 循环 step 级 1 直接 `wp.copy` |
| `positions_hat_np` | 惯性预测位置 x̂ = x^n + Δt·v^n + Δt²·M⁻¹·f_ext | ndarray (n,3) float64 | main.py step 级 2 |
| `positions_hat_wp` | 惯性位置 x̂ 的 GPU 形态（动能 kernel 锚点） | wp.array vec3 (n,) | newton_solver.set_hat_position() |
| `positions_guess_np` | Newton 内层迭代的猜测位置（= x̂ 起步，逐步更新） | ndarray (n,3) float64 | main.py step 级 4 / 迭代 10 |
| `positions_rest_np` | 参考/初始位置（rest pose，固定顶点永远回到这里） | ndarray (n,3) float64 | main.py 初始化 |
| `velocities_np` | 顶点速度 v^n（隐式欧拉 v = (x* − x^n)/Δt 更新） | ndarray (n,3) float64 | main.py 循环 |
| `masses_np` / `masses_wp` | 顶点质量（每个 tet 质量 rho·vol 均分给 4 顶点） | ndarray (n,) / wp.array float | main.py / newton_solver |
| `pinned_dof_mask` | 固定自由度掩码 (3n,)（true 表示该 DOF 不动） | ndarray (3n,) bool | main.py → solver.free_mask |
| `free_vertices_mask` | 非固定顶点掩码 (n,1)，numpy 广播用 | ndarray (n,1) bool | main.py |

## 2. 四面体拓扑与参考几何

| 变量 | 含义 | 类型 / 形状 | 位置 |
|---|---|---|---|
| `verts_np` / `tets_np` | 初始网格顶点 / 四面体索引 | ndarray (n,3) / (nt,4) int32 | main.py 初始化 |
| `tet_indices` | 四面体索引的 GPU 形态 | wp.array vec4i (nt,) | main.py / solver |
| `Dm_rest` | 参考边矩阵 Dm = [x1−x0, x2−x0, x3−x0]（rest 构型） | wp.array mat33 (nt,) | main.py 初始化 4 |
| `Dm_rest_inv` | 参考边矩阵的逆（应变能实际用它：F = Ds·Dm_inv） | wp.array mat33 (nt,) | main.py → solver.Dm_inv |
| `rest_volumes` / `rest_volumes_np` | 参考体积（每个 tet 的能量 = phi·rest_volume） | wp.array float / ndarray (nt,) | main.py 初始化 4 |
| 应变能中间量 `Ds` | 当前边矩阵（kernel 输出） | wp.array mat33 (nt,) | newton_solver |
| 应变能中间量 `F` | 变形梯度 F = Ds·Dm_rest_inv | wp.array mat33 (nt,) | newton_solver |
| `tet_energies` | 每个 tet 的应变能（# 逐个 tet 计算再归约） | wp.array float (nt,) | newton_solver |

## 3. 接触候选对与 lagged 摩擦数据

| 变量 | 含义 | 类型 / 形状 | 位置 |
|---|---|---|---|
| `surface_faces_np` / `surface_edges_np` | 去重后的外表面三角形 / 边（内部面出现 2 次被剔除） | ndarray | main.py 初始化 1 |
| `PT_pair` | 点-三角形候选对（表面顶点 × 不含它的面） | wp.array vec4i | main.py 初始化 2+3 |
| `EE_pair` | 边-边候选对（无共享顶点的两条边） | wp.array vec4i | main.py 初始化 2+3 |
| `lagged_data` | 摩擦滞后数据包，Newton 期间只读：`PT_beta`（β 重心坐标）、`PT_tangent0/1`（切平面基）、`PT_normal_force`（法向力 λ）、`EE_alpha`（边参数）、`EE_tangent0/1`、`EE_normal_force` | dataclass FrictionLaggedData | main.py step 级 3 |
| `pt_distances_squared` / `ee_distances_squared` | 各候选对当前距离平方（CCD / barrier 检查用） | ndarray (各对,) | newton_solver._pair_distances |

## 4. 物理常数与求解常数

| 变量 | 含义 | 值（demo） |
|---|---|---|
| `DT` | 时间步长 | 0.01 |
| `NUM_STEPS` | 总时间步数 | 8 |
| `GRAVITY` / `external_forces` | 重力加速度 / 每顶点外力 f_ext | (0,0,−9.8) |
| `d_tilde` | Barrier 激活阈值（距离 < 它认为接触） | 1e-3 |
| `kappa` | Barrier 强度参数 | 1e5 |
| `E` | 杨氏模量 | 1e5 |
| `NIU` | 泊松比 | 0.1 |
| `mu` / `lamda` | Lame 系数（由 E、NIU 派生） | — |
| `friction_mu` | 摩擦系数 | 0.3 |
| `eps_v` / `y_eps` | 摩擦光滑化速度阈值 / 位移分段点 = DT·eps_v | 1e-3 / 1e-5 |
| `rho` | 密度（质量均分到顶点） | 1000.0 |
| `GRADIENT_TOLERANCE` | 收敛：梯度范数阈值（相对初始） | 1e-6 |
| `STEP_TOLERANCE` | 收敛：步长范数阈值 | 1e-8 |
| `MAX_NEWTON_ITER` | 内层 Newton 最大迭代数 | 15 |
| `GRID_CELLS` / `CUBE_SIZE` / `INITIAL_GAP` / `LATERAL_OFFSET` | demo 网格参数（cell 数 / 边长 / 间隙 / 偏移） | 2 / 0.2 / 0.01 / 0.02 |

## 5. Newton 迭代局部量（main.py 循环内）

| 变量 | 含义 |
|---|---|
| `potential_energy` | 在猜测位置求的总势能 E(x)（evaluate 返回的标量） |
| `gradient_np` | 总能量梯度 g = ∇E(x)，(n,3)，残留力 |
| `energy_initial` | 首个 Newton 迭代点的能量 E0（报告用） |
| `initial_gradient_norm` | 首个迭代的自由梯度范数 ‖g0‖∞（相对收敛阈值基准） |
| `free_gradient_norm` | 当前自由度梯度最大分量 max\|g[free]\|（收敛判据 1） |
| `hessian_free` | FD Hessian 的自由子矩阵 H_ff (Nf,Nf)（迭代 6） |
| `free_dof_indices` | 自由自由度下标（Hessian 只算这些列；固定 DOF 从系统剔除） |
| `direction_np` | Newton 方向 p（解 H·p = −g 得到，迭代 7） |
| `alpha_max` | CCD 允许的最大步长（迭代 8） |
| `alpha` | Armijo 决定的最终步长（迭代 9） |
| `update_max_norm` | 实际更新量 max\|α·p\|（收敛判据 2） |
| `energy_final` / `gradient_final` | step 收敛后的能量与梯度（报告用） |
| `min_pair_distance` | 所有候选对最近距离 √min(d²)（确认 barrier 激活用） |

## 6. newton_solver 内部方法参数

| 参数 | 含义 |
|---|---|
| `positions_np` | 各种求值/检查方法入参：numpy 猜测位置 |
| `energy_at_x` | Armijo 的当前能量 E(x_k)（传入即 energy） |
| `gradient_np` / `direction_np` | Armijo / CCD 入参：梯度、方向 p |
| `hessian_free` / `free_dof_indices` | compute_newton_direction 入参：H_ff 与自由下标 |
| `fd_step_scale` / `fd_step` | FD 差分步长系数 / 实际步长 h = 系数·(1+‖x‖∞) |
| `sufficient_decrease` | Armijo 充分下降系数 c = 1e-4 |
| `alpha_lower` / `alpha_upper` / `alpha_mid` | CCD 二分的 lo / hi / mid |
| `gradient_free` / `direction_free` | 自由子空间上的梯度块 / 方向块（solve 对象） |
| `gradient_plus` / `gradient_minus` / `positions_plus` / `positions_minus` | FD 中心差分的 +h / −h 侧梯度与位置 |

## 7. Warp kernel 内部记号（几何惯例，属于底层工具库）

| 记号 | 含义 |
|---|---|
| `A, B, C（D）` | 三角形 / 边的四个顶点（点-三角形、边-边距离） |
| `e0, e1` | 三角形或边的边向量 |
| `point` | PT 对里的顶点（vs 三角形 ABC） |
| `x0..x3` / `i0..i3` | 四面体的 4 个顶点 / 索引 |
| `D`（kernel 内） | 边矩阵 matrix_from_cols(...) |
| `tet_index` | 当前四面体线程下标 wp.tid() |

---

### 伪代码 ↔ 代码对照速查

| 伪代码概念 | 代码变量 |
|---|---|
| 滞后位置 | `positions_lagged_np` / `positions_lagged_wp` |
| 惯性位置 | `positions_hat_np` / `positions_hat_wp` |
| 猜测位置 | `positions_guess_np`（写入 `positions_wp` 求梯度） |
| 初始位置 | `positions_rest_np` |
| lagged 信息（β/α、切平面、λ） | `lagged_data`（`FrictionLaggedData`） |
| 迭代方向 p | `direction_np` |
| 最大允许步长 | `alpha_max`（CCD） |
| 最终步长 | `alpha`（Armijo） |