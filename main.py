"""IPC Lite 教学版主流程 ---- 与学习笔记伪代码一一对应.

伪代码(教学重写版,与 IPC 原论文算法流程一致):

    物理常数: DT, NUM_STEPS, gravity, d_tilde, kappa, E, NIU, mu, lamda,
              friction_mu, eps_v, y_eps

    初始化:
      1. 提取所有 tetra 的表面(得到表面顶点/面/边)
      2. 形成 PT,EE 候选对(可能碰撞/摩擦的顶点-面,边-边组合)
      3. 剔除初始距离严格为 0 的退化候选对(相邻共面产生,barrier 在 d=0 发散)
      4. 计算参考边矩阵 Dm_rest,其逆 Dm_rest_inv,参考体积 rest_volumes

    for step in NUM_STEPS:
      1. 更新滞后位置 positions_lagged_np ← 上一时刻解 x^n
      2. 更新惯性位置 positions_hat_np = x^n + Δt·v^n + Δt²·M⁻¹·external_forces
      3. 根据滞后位置计算 lagged 信息(β/α 接触点参数,切平面,法向力 λ)
      4. 将惯性位置作为初始猜测位置 positions_guess_np = x̂
         for iteration in NUM_ITERATION:
           1. 计算动能(惯性势能):需要 猜测位置,惯性位置
           2. 计算弹性能:需要 猜测位置
           3. 计算 Barrier 接触能:需要 接触候选对,猜测位置
           4. 计算摩擦能:需要 接触候选对,猜测位置,滞后位置
           5. 计算能量和
           6. 计算梯度 g = ∇E 以及 Hessian H = ∇²E(FD 差分)
           7. 使用 Newton 法确定迭代方向 p:H·p = −g
           8. 通过 CCD 确定最大允许步长 α_max
           9. 使用 Armijo 条件决定最终步长 α
          10. 更新猜测位置 x ← x + α·p
          11. 收敛判断(梯度范数 / 步长 / 迭代上限)
      5. 使用猜测位置(收敛解 x*)和滞后位置计算速度 v^{n+1} = (x* − x^n)/Δt

demo:上下两块立方体,下块底层固定(Dirichlet),上块自由落体砸向下块,
碰撞时 barrier 激活.初始速度 0(摩擦在此场景下几乎不激活,属正常现象).

性能说明:FD Hessian 每个自由度要 2 次 GPU 能量求值,只适合小网格教学;
真实 IPC 用解析 Hessian + 稀疏求解(替换点在 newton_solver.compute_hessian_fd).
"""

import numpy as np
import warp as wp

import geometry
import energy
from mass import compute_lumped_masses
from newton_solver import NewtonSolver

device = "cuda:0" if wp.is_cuda_available() else "cpu"

# ======================================================================
# 物理常数(伪代码:物理常数)
# ======================================================================

# ---- 应变能(stable neo-Hookean)----
NIU = 0.1                       # 泊松比
E = 1e5                         # 杨氏模量
mu = E / (2 * (1 + NIU))        # 剪切模量
lamda = E * NIU / ((1 + NIU) * (1 - 2 * NIU))   # Lame 第 2 参数

# ---- Barrier 接触 ----
d_tilde = 1e-3                  # Barrier 激活阈值:距离小于它认为"发生接触"
kappa = 1e5                     # Barrier 强度参数

# ---- 时间与求解 ----
DT = 0.01                       # 时间步长
NUM_STEPS = 8                   # 总时间步数(伪代码: NUM_STEPS)
MAX_NEWTON_ITER = 15            # 内层 Newton 最大迭代数
GRADIENT_TOLERANCE = 1e-6                    # 收敛:梯度范数阈值(相对初始梯度)
STEP_TOLERANCE = 1e-8                    # 收敛:步长范数阈值

# ---- 摩擦 ----
friction_mu = 0.3               # 摩擦系数
eps_v = 1e-3                    # 摩擦光滑化速度阈值
y_eps = DT * eps_v              # f0 光滑化分段点(速度阈值 × 时间步长)

# ---- 质量与外力 ----
rho = 1000.0                    # 密度(质量均匀分散到顶点)
GRAVITY = np.array([0.0, 0.0, -9.8])   # 重力加速度

# ---- demo 网格参数 ----
GRID_CELLS = 2                          # 每轴 cell 数(单块 27 顶点 / 48 tets)
CUBE_SIZE = 0.2                      # 立方体边长
INITIAL_GAP = 0.01                      # 两块初始间隙(自由落体约 4.5 步后接触)
LATERAL_OFFSET = 0.02                  # 上块横向偏移

# ======================================================================
# 状态变量分配(伪代码变量清单)
# ======================================================================

# ---- 网格 ----
verts_np, tets_np, pinned_np = geometry.build_two_cubes(GRID_CELLS, CUBE_SIZE, INITIAL_GAP, LATERAL_OFFSET)
num_vertices = verts_np.shape[0]
num_tets = tets_np.shape[0]
print(f"网格: {num_vertices} 顶点, {num_tets} tets, "
      f"固定 {pinned_np.sum()} 顶点(下块底层)")

# 固定自由度掩码(下块最底层,Dirichlet 边界):
# pinned_dof_mask 按 x/y/z 展开成 (3n,) 给 solver;free_vertices_mask 是 (n,1) 给 numpy 广播.
pinned_dof_mask = np.repeat(pinned_np, 3)
free_vertices_mask = ~pinned_np[:, None]

# 顶点初始位置(= 参考位置,t=0 的状态)
positions_rest_np = verts_np.astype(np.float64)   # numpy (n,3) float64

# 顶点位置:整个模拟的状态本体,Python 侧用 numpy float64 维护.
# 注意与 solver.positions_wp(warp float32 可求梯度)的区分:
# numpy 无梯度,只能当"账本";kernel 计算必须用 warp 数组.
positions_np = positions_rest_np.copy()
velocities_np = np.zeros((num_vertices, 3), dtype=np.float64)
# 顶点质量在"初始化 4"算完参考体积后填充(lumped:每个 tet 质量均分给 4 个顶点)

# ---- 四面体参考几何(伪代码初始化 4)----
tet_indices = wp.array(tets_np, dtype=wp.vec4i, device=device)
Dm_rest = wp.empty(num_tets, dtype=wp.mat33, device=device)        # 参考边矩阵
Dm_rest_inv = wp.empty(num_tets, dtype=wp.mat33, device=device)    # 参考边矩阵的逆
rest_volumes = wp.empty(num_tets, dtype=float, device=device)      # 参考体积

# ======================================================================
# 初始化(伪代码:初始化 1~4)
# ======================================================================

print("===== 初始化 =====")

# ---- 初始化 1:提取表面 ----
# 所有 tet 的面写出后去重:内部面出现 2 次,只保留出现 1 次的外表面
all_faces = wp.empty(num_tets * 4, dtype=wp.vec3i, device=device)
wp.launch(
    kernel=geometry.extract_all_faces,
    dim=num_tets,
    inputs=[tet_indices],
    outputs=[all_faces],
    device=device,
)
surface_faces, surface_edges = geometry.extract_surface_faces_and_edges(all_faces)
surface_faces_np = surface_faces.numpy()
surface_edges_np = surface_edges.numpy()

# ---- 初始化 4:参考边矩阵,逆,参考体积 ----
wp.launch(
    kernel=energy.compute_rest_data,
    dim=num_tets,
    inputs=[tet_indices, wp.array(verts_np, dtype=wp.vec3, device=device)],
    outputs=[Dm_rest, Dm_rest_inv, rest_volumes],
    device=device,
)
rest_volumes_np = rest_volumes.numpy()

# ---- 质量(放到参考体积之后)----
masses_np = compute_lumped_masses(tets_np, rest_volumes_np, num_vertices, rho)

# ---- 初始化 2+3:形成 PT/EE 候选对并剔除退化对 ----
# 2:暴力枚举表面顶点×三角形,边×边(固定全集,教学简化)
# 3:剔除初始距离严格为 0 的退化对(同一表面相邻共面的三角形之间会
#    出现"顶点投影落在共享边上"距离=0 的对,barrier 在 d=0 发散,必删)
#    注意不能用 d_tilde 过滤:那会把初始就接近的合法接触对删掉,
#    它们将永远无法再激活接触.
PT_pair, EE_pair = geometry.filter_degenerate_candidates(
    surface_faces_np,
    surface_edges_np,
    verts_np,
    threshold_sq=1e-12,
    device=device,
)
print(f"候选对: PT={PT_pair.shape[0]}, EE={EE_pair.shape[0]}")

# ======================================================================
# Newton 求解器(持有 Warp 数组,tape,能量 kernel 装配)
# ======================================================================

positions_wp = wp.from_numpy(
    verts_np, dtype=wp.vec3, device=device, requires_grad=True
)   # 猜测位置的 warp 形态(可求梯度)

solver = NewtonSolver(
    device=device,
    num_vertices=num_vertices,
    num_tets=num_tets,
    positions_wp=positions_wp,
    tet_indices=tet_indices,
    Dm_inv=Dm_rest_inv,
    rest_volumes=rest_volumes,
    mu=mu,
    lamda=lamda,
    PT_pair=PT_pair,
    EE_pair=EE_pair,
    d_tilde=d_tilde,
    kappa=kappa,
    friction_mu=friction_mu,
    y_eps=y_eps,
    masses_np=masses_np,
    dt=DT,
    pinned_dof_mask=pinned_dof_mask,
)

# ======================================================================
# 时间步循环(伪代码:for step in NUM_STEPS)
# ======================================================================

for step in range(NUM_STEPS):
    print(f"\n===== step {step} =====")

    # ================================================================
    # 伪代码 step 级 1:更新滞后位置
    # ================================================================
    # positions_wp 此刻 = 上一步收敛解 x^n,把它冻结成滞后构型.
    # Newton 迭代会不断改写 positions_wp,必须留这份不动的副本给
    # 摩擦相对位移 Δ = x − x^n 以及 λ/β/α(lagged 数据)使用.
    wp.copy(solver.positions_lagged_wp, solver.positions_wp)
    positions_lagged_np = positions_np.copy()      # numpy 副本(速度更新用)

    # ================================================================
    # 伪代码 step 级 2:更新惯性位置 x̂ = x^n + Δt·v^n + Δt²·M⁻¹·external_forces
    # ================================================================
    external_forces = masses_np[:, None] * GRAVITY
    positions_hat_np = (
        positions_lagged_np
        + DT * velocities_np
        + (DT * DT) * (external_forces / masses_np[:, None])
    )
    # 固定顶点永远留在原位
    positions_hat_np = np.where(free_vertices_mask, positions_hat_np, positions_rest_np)
    solver.set_hat_position(positions_hat_np)

    # ================================================================
    # 伪代码 step 级 3:根据滞后位置计算 lagged 摩擦信息
    # ================================================================
    # 在 x^n 上算好并冻结整个 Newton 期间:
    #   - β/α:接触点参数化(最近点的重心坐标)→ 定义滑动位移算子 Γ
    #   - tangent0/1:接触切平面基 → 把相对位移投影到切平面
    #   - 法向力 λ = barrier 力在 x^n 上的值 → 摩擦强度
    lagged_data = energy.compute_friction_lagged_data(
        positions_lagged=solver.positions_lagged_wp,
        PT_pair=PT_pair,
        EE_pair=EE_pair,
        d_tilde=d_tilde,
        kappa=kappa,
    )
    solver.set_lagged_data(lagged_data)

    # ================================================================
    # 伪代码 step 级 4:将惯性位置作为初始猜测位置
    # ================================================================
    positions_guess_np = positions_hat_np.copy()

    # ---- 收敛记录 ----
    initial_gradient_norm = 1.0
    converged = False
    energy_initial = float("nan")

    # ================================================================
    # 内层 Newton 迭代(伪代码:for iteration in NUM_ITERATION)
    # ================================================================
    for iteration in range(MAX_NEWTON_ITER):

        # ---- 伪代码 1~5 + 6:算四项能量,能量和,梯度 ----
        # 一次前向+反向同时拿到 总能量 E 和 梯度 g = ∇E(猜测位置).
        # 四项能量在 solver._launch_all_energies 里按 1~5 编号装配.
        potential_energy, gradient_np = solver.evaluate_energy_and_gradient(positions_guess_np)
        if iteration == 0:
            energy_initial = potential_energy
            initial_gradient_norm = float(np.max(np.abs(gradient_np.reshape(-1)[solver.free_dof_indices])))

        # ---- 伪代码 11(第一处):收敛判断 ---- 梯度范数 ----
        # 梯度 = 每个顶点的残留力;梯度小 = 力平衡 = 到达极小点.
        # 只统计自由自由度:固定顶点的支撑反力永远不归零.
        free_gradient_norm = float(np.max(np.abs(gradient_np.reshape(-1)[solver.free_dof_indices])))
        if free_gradient_norm < GRADIENT_TOLERANCE * (1.0 + initial_gradient_norm):
            converged = True
            print(f"    Newton iter {iteration}: |g|={free_gradient_norm:.3e} 收敛(梯度)")
            break

        # ---- 伪代码 6(二阶):Hessian(中心差分)----
        # 只算自由自由度列(固定列不需要).FD 噪声见模块注释.
        hessian_free, free_dof_indices = solver.compute_hessian_fd(positions_guess_np)

        # ---- 伪代码 7:Newton 方向 ----
        # 解 (H + M/Δt²)·p = −g.动能项 M/Δt² 已包含在 FD 的 H 里.
        direction_np = solver.compute_newton_direction(hessian_free, gradient_np, free_dof_indices)

        # ---- 伪代码 8:CCD 确定最大允许步长 α_max ----
        alpha_max = solver.ccd_alpha_max(positions_guess_np, direction_np)
        if alpha_max <= 0.0:
            print(f"    Newton iter {iteration}: CCD 无可行步,提前结束")
            break

        # ---- 伪代码 9:Armijo 决定最终步长 α ----
        alpha = solver.armijo_line_search(
            positions_guess_np, potential_energy, gradient_np, direction_np, alpha_max
        )
        if alpha <= 0.0:
            print(f"    Newton iter {iteration}: Armijo 失败,提前结束")
            break

        # ---- 伪代码 10:更新猜测位置 ----
        update_max_norm = float(np.max(np.abs(alpha * direction_np)))
        positions_guess_np = positions_guess_np + alpha * direction_np.reshape(-1, 3)
        print(
            f"    Newton iter {iteration}: E={potential_energy:.6e} |g|={free_gradient_norm:.3e} "
            f"alpha={alpha:.3e} step={update_max_norm:.3e}"
        )

        # ---- 伪代码 11(第二处):收敛判断 ---- 步长范数 ----
        if update_max_norm < STEP_TOLERANCE:
            converged = True
            print(f"    Newton iter {iteration}: 步长 < {STEP_TOLERANCE} 收敛")
            break

    # ================================================================
    # 伪代码 step 级 5:速度更新
    # ================================================================
    # 收敛后的猜测位置 = 本步解 x*;速度 = (x* − x^n)/Δt(隐式欧拉).
    velocities_np = np.where(
        free_vertices_mask, (positions_guess_np - positions_lagged_np) / DT, 0.0
    )
    positions_np = positions_guess_np
    wp.copy(
        positions_wp,
        wp.from_numpy(
            positions_np.astype(np.float32), dtype=wp.vec3, device=device
        ),
    )

    # ---- 本步报告 ----
    energy_final, gradient_final = solver.evaluate_energy_and_gradient(positions_np)
    pt_distances_squared, ee_distances_squared = solver._pair_distances(positions_np)
    min_pair_distance = (np.concatenate([pt_distances_squared, ee_distances_squared]).min() ** 0.5) if pt_distances_squared.size + ee_distances_squared.size else 0.0
    print(
        f"[step {step} 结果] 迭代={iteration + 1} 收敛={converged} "
        f"E: {energy_initial:.6e} -> {energy_final:.6e}, "
        f"|g_free|: {initial_gradient_norm:.3e} -> "
        f"{np.max(np.abs(gradient_final.reshape(-1)[solver.free_dof_indices])):.3e}"
    )
    print(f"          最大位移 = {np.max(np.abs(positions_np - positions_rest_np)):.4e}")
    print(f"          最近接触距离 min_pair_distance = {min_pair_distance:.6e} (barrier 阈值 d_tilde={d_tilde:.1e})")