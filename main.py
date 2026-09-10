"""IPC 学习版主流程。

一个时间步 = 惯性预测 + lagged 摩擦数据 + Newton 求解(梯度/FD Hessian/线搜索) + 速度更新。

运行的是 2x2x2 小正方体网格（81 个自由度），顶部给一个向下的速度压向
底部固定层，同时带横向滑移（激发 barrier 和 friction 两类能量）。
FD Hessian 开销随自由度线性增长，demo 网格小才跑得动；大网格请换解析 Hessian。
"""

import numpy as np
import warp as wp

import geometry
import energy
from mass import compute_lumped_masses
from newton_solver import NewtonSolver

device = "cuda:0"

# ---------------- 物理参数 ----------------
# Strain energy constants
NIU = 0.1
E = 1e5

# Lame constants
mu = E / (2 * (1 + NIU))
lamda = E * NIU / ((1 + NIU) * (1 - 2 * NIU))

# Barrier constant
d_tilde = 1e-3
kappa = 1e5

# 时间步长
DT = 0.01

# Friction constant
friction_mu = 0.3
eps_v = 1e-3
y_eps = DT * eps_v

# 密度
rho = 1000.0
GRAVITY = np.array([0.0, 0.0, -9.8])

# ---------------- demo 网格参数 ----------------
NC = 2          # 每轴 cell 数（单块 2 -> 27 顶点 / 48 tets）
SIZE = 0.2      # 边长
GAP = 0.03      # 两块之间初始间隙（> d_tilde，接近时 barrier 激活）
LATERAL = 0.02  # 上块的横向偏移（碰撞时激发摩擦滑动）
NUM_STEPS = 8   # 时间步数（FD Hessian 偏慢，先跑几步验证）
MAX_NEWTON_ITER = 15
TOL_G = 1e-6
TOL_X = 1e-8

# ---------------- 网格生成（上下两块立方体，底部固定） ----------------
verts_np, tets_np, pinned_np = geometry.build_two_cubes(NC, SIZE, GAP, LATERAL)
num_vertices = verts_np.shape[0]
num_tets = tets_np.shape[0]
pinned_flat = np.repeat(pinned_np, 3)          # (3n,) bool
free2d = ~pinned_np[:, None]                   # (n,1) bool，用于 numpy 广播

print(f"网格: {num_vertices} 顶点, {num_tets} tets, 固定 {pinned_np.sum()} 顶点")

rest_positions = wp.array(verts_np, dtype=wp.vec3, device=device)
positions = wp.from_numpy(verts_np, dtype=wp.vec3, device=device, requires_grad=True)
positions_prev = wp.empty(num_vertices, dtype=wp.vec3, device=device)
tet_indices = wp.array(tets_np, dtype=wp.vec4i, device=device)

# 参考数据
Dm = wp.empty(num_tets, dtype=wp.mat33, device=device)
Dm_inv = wp.empty(num_tets, dtype=wp.mat33, device=device)
rest_volumes = wp.empty(num_tets, dtype=float, device=device)
wp.launch(
    kernel=energy.compute_rest_data,
    dim=num_tets,
    inputs=[tet_indices, rest_positions],
    outputs=[Dm, Dm_inv, rest_volumes],
    device=device,
)
rest_volumes_np = rest_volumes.numpy()

# ---------------- 表面与接触候选对 ----------------
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

# 固定候选集合：去掉初始距离 < d_tilde 的退化对（同表面相邻共面对）
# 固定候选集合：只剔除距离严格为 0 的退化对（同表面相邻共面对）
PT_pair, EE_pair = geometry.filter_degenerate_candidates(
    surface_faces_np,
    surface_edges_np,
    verts_np,
    threshold_sq=1e-12,
    device=device,
)
print(f"候选对: PT={PT_pair.shape[0]}, EE={EE_pair.shape[0]}")

# ---------------- 质量矩阵 ----------------
masses_np = compute_lumped_masses(tets_np, rest_volumes_np, num_vertices, rho)

# ---------------- Newton 求解器 ----------------
solver = NewtonSolver(
    device=device,
    num_vertices=num_vertices,
    num_tets=num_tets,
    positions=positions,
    tet_indices=tet_indices,
    Dm_inv=Dm_inv,
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
    pinned_flat=pinned_flat,
)

# ---------------- 初始状态 ----------------
# 上块整体向下落 + 横向滑移，砸向固定的下块：碰撞时 barrier 激活，
# 横向相对运动激发摩擦。
n_bottom = num_vertices // 2   # 下块顶点数（两块等大）
velocity_np = np.zeros((num_vertices, 3), dtype=np.float64)
velocity_np[n_bottom:, 0] = 1.0    # 横向滑移 → 摩擦
velocity_np[n_bottom:, 2] = -1.0   # 向下落（dt*v=0.01 < 间隙 0.03，x_hat 不穿透）→ barrier

positions_np = verts_np.astype(np.float64).copy()

# =====================================================================
# 时间步循环
# =====================================================================
for step in range(NUM_STEPS):
    print(f"\n===== step {step} =====")

    x_prev = positions_np.copy()
    v_prev = velocity_np.copy()

    '''惯性预测位置 x_hat = x^n + dt*v^n + dt^2 * M^-1 * f_ext'''
    f_ext = masses_np[:, None] * GRAVITY
    x_hat = x_prev + DT * v_prev + (DT * DT) * (f_ext / masses_np[:, None])
    # 固定顶点永远留在原位
    x_hat = np.where(free2d, x_hat, verts_np)

    '''冻结上一时间步位置 x^n，作为摩擦的滞后构型'''
    wp.copy(positions_prev, positions)

    '''计算摩擦滞后数据（beta/alpha、切向基、法向力 N），Newton 期间冻结'''
    lagged_data = energy.compute_friction_lagged_data(
        positions_lagged=positions_prev,
        PT_pair=PT_pair,
        EE_pair=EE_pair,
        d_tilde=d_tilde,
        kappa=kappa,
    )

    solver.set_step(x_hat, positions_prev, lagged_data)

    '''Newton 求解：梯度 -> FD Hessian -> CCD -> Armijo -> 更新 -> 收敛判断'''
    x_new, info = solver.solve_step(
        x_hat,
        max_iter=MAX_NEWTON_ITER,
        tol_g=TOL_G,
        tol_x=TOL_X,
        verbose=True,
    )

    '''速度更新：v^{n+1} = (x^{n+1} - x^n) / dt，固定点速度清零'''
    velocity_np = np.where(free2d, (x_new - x_prev) / DT, 0.0)
    positions_np = x_new
    wp.copy(positions, wp.from_numpy(x_new.astype(np.float32), dtype=wp.vec3, device=device))

    print(
        f"[step {step} 结果] 迭代={info['iterations']} 收敛={info['converged']} "
        f"E: {info['E0']:.6e} -> {info['Efinal']:.6e}, "
        f"|g_free|: {info['grad_norm0']:.3e} -> {info['grad_norm1']:.3e}"
    )
    print(f"          最大位移 = {np.max(np.abs(positions_np - verts_np)):.4e}")
    # 最近接触距离（确认 barrier 是否进入激活区 d < d_tilde）
    pt_d2, ee_d2 = solver._pair_distances(positions_np)
    d_min = (np.concatenate([pt_d2, ee_d2]).min() ** 0.5) if pt_d2.size + ee_d2.size else 0.0
    print(f"          最近接触距离 d_min = {d_min:.6e} (barrier 激活阈值 d_tilde={d_tilde:.1e})")