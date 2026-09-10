"""IPC 的 Newton 求解器（学习版）。

把 main.py 里的能量装配封装成可复用的求值函数，并补全 Newton 迭代的
完整链路：

    能量求值 -> 梯度(Warp AD) -> Hessian(有限差分) -> 线性求解
    -> CCD 步长过滤 -> Armijo 线搜索 -> 更新 x -> 收敛判断

为什么要用有限差分算 Hessian：
    Warp 的 tape 只支持一阶反向自动微分，二阶导需要自己算。
    这里用中心差分：H[:, j] = (g(x + h*e_j) - g(x - h*e_j)) / (2h)，
    每个自由度 2 次梯度求值。demo 网格只有几十个顶点，开销可接受；
    网格变大了记得换成解析 Hessian（那是 IPC 正规做法）。

CCD 过滤是简化版：在 [0,1] 上对 alpha 做二分，检验候选位置是否
满足 (1) 所有四面体体积 > 0，(2) 没有任何接触对真实穿透(d^2 接近 0)。
真实 IPC 用精确 CCD(时间连续碰撞检测)求每个对的最大步长再取 min，
这里用二分采样近似，学习用途足够。
"""

import numpy as np
import warp as wp

import energy
import geometry
from mass import compute_kinetic_energy


@wp.kernel
def signed_tet_volumes_kernel(
    tet_indices: wp.array[wp.vec4i],
    positions: wp.array[wp.vec3],
    volumes: wp.array[float],
):
    t = wp.tid()
    tet = tet_indices[t]
    x0 = positions[tet[0]]
    x1 = positions[tet[1]]
    x2 = positions[tet[2]]
    x3 = positions[tet[3]]
    D = wp.matrix_from_cols(x1 - x0, x2 - x0, x3 - x0)
    volumes[t] = wp.determinant(D) / 6.0


class NewtonSolver:
    """持有全部能量数组与 tape，提供 IPC 单步求解。"""

    def __init__(
        self,
        device,
        num_vertices,
        num_tets,
        positions,          # wp.array vec3, requires_grad=True
        tet_indices,        # wp.array vec4i
        Dm_inv,             # wp.array mat33
        rest_volumes,       # wp.array float
        mu,
        lamda,
        PT_pair,            # wp.array vec4i
        EE_pair,            # wp.array vec4i
        d_tilde,
        kappa,
        friction_mu,
        y_eps,
        masses_np,          # numpy (num_vertices,) lumped mass
        dt,
        pinned_flat=None,   # numpy bool (3*num_vertices,) 或 None
    ):
        self.device = device
        self.num_vertices = num_vertices
        self.num_tets = num_tets

        self.positions = positions
        self.tet_indices = tet_indices
        self.Dm_inv = Dm_inv
        self.rest_volumes = rest_volumes
        self.mu = mu
        self.lamda = lamda
        self.PT_pair = PT_pair
        self.EE_pair = EE_pair
        self.d_tilde = d_tilde
        self.kappa = kappa
        self.friction_mu = friction_mu
        self.y_eps = y_eps
        self.dt = dt

        self.masses_np = np.asarray(masses_np, dtype=np.float64).reshape(-1)
        self.masses_wp = wp.array(self.masses_np.astype(np.float32), device=device)

        self.num_dofs = 3 * num_vertices
        if pinned_flat is None:
            pinned_flat = np.zeros(self.num_dofs, dtype=bool)
        self.free_mask = ~np.asarray(pinned_flat, dtype=bool).reshape(-1)

        # 每个 step 更新一次的惯性预测位置（动能项用）
        self.x_hat_wp = wp.empty(num_vertices, dtype=wp.vec3, device=device)

        # 应变能工作数组（compute_spatial_data 的输出）
        self.Ds = wp.empty(num_tets, dtype=wp.mat33, device=device)
        self.F = wp.empty(num_tets, dtype=wp.mat33, device=device)
        self.tet_energies = wp.empty(num_tets, dtype=float, device=device, requires_grad=True)

        # 各能量归约目标
        self.kinetic_total = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        self.elastic_total = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        self.PT_total = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        self.EE_total = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        self.friction_total = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        self.total_energy = wp.zeros(1, dtype=float, device=device, requires_grad=True)

        self.PT_barrier_energies = wp.empty(
            PT_pair.shape[0], dtype=float, device=device, requires_grad=True
        )
        self.EE_barrier_energies = wp.empty(
            EE_pair.shape[0], dtype=float, device=device, requires_grad=True
        )
        self.PT_friction_energies = wp.empty(
            PT_pair.shape[0], dtype=float, device=device, requires_grad=True
        )
        self.EE_friction_energies = wp.empty(
            EE_pair.shape[0], dtype=float, device=device, requires_grad=True
        )

        self.tet_volumes = wp.empty(num_tets, dtype=float, device=device)
        self.tape = wp.Tape()

        # 摩擦滞后量：每个 step 由 solve_step 注入，Newton 期间冻结
        self.lagged_data = None

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _set_positions(self, x_np):
        """把 numpy 位置写入 Warp positions 数组（float32）。"""
        wp.copy(
            self.positions,
            wp.from_numpy(
                np.ascontiguousarray(x_np, dtype=np.float32),
                dtype=wp.vec3,
                device=self.device,
            ),
        )

    def _zero_targets(self):
        """归约目标清零（reduce_energy 用 atomic_add，不清会跨次累加）。"""
        self.kinetic_total.zero_()
        self.elastic_total.zero_()
        self.PT_total.zero_()
        self.EE_total.zero_()
        self.friction_total.zero_()
        self.total_energy.zero_()

    # ------------------------------------------------------------------
    # 能量求值
    # ------------------------------------------------------------------

    def _forward(self, x_np, record_tape):
        """前向算全部能量（动能 + 弹性能 + barrier + 摩擦），写 total_energy。"""
        self._set_positions(x_np)
        self._zero_targets()

        launch = wp.launch
        with record_tape:

            # 动能项（惯性势能），位置和 x_hat 的差在 dt^2 尺度上
            launch(
                compute_kinetic_energy,
                dim=self.num_vertices,
                inputs=[self.positions, self.x_hat_wp, self.masses_wp, self.dt],
                outputs=[self.kinetic_total],
                device=self.device,
            )

            # 弹性能
            launch(
                energy.compute_spatial_data,
                dim=self.num_tets,
                inputs=[
                    self.tet_indices,
                    self.positions,
                    self.Dm_inv,
                    self.rest_volumes,
                    self.mu,
                    self.lamda,
                ],
                outputs=[self.Ds, self.F, self.tet_energies],
                device=self.device,
            )

            # Barrier 接触能
            energy.compute_barrier_energies(
                self.positions,
                self.PT_pair,
                self.EE_pair,
                self.d_tilde,
                self.kappa,
                self.PT_barrier_energies,
                self.EE_barrier_energies,
            )

            # 摩擦耗散能（lagged_data 冻结）
            energy.compute_friction_energies(
                self.positions,
                self.lagged_positions,
                self.PT_pair,
                self.EE_pair,
                self.lagged_data,
                self.friction_mu,
                self.y_eps,
                self.PT_friction_energies,
                self.EE_friction_energies,
            )

            # 归约 + 合并
            launch(
                energy.reduce_energy,
                dim=self.num_tets,
                inputs=[self.tet_energies],
                outputs=[self.elastic_total],
                device=self.device,
            )
            if self.PT_pair.shape[0] > 0:
                launch(
                    energy.reduce_energy,
                    dim=self.PT_pair.shape[0],
                    inputs=[self.PT_barrier_energies],
                    outputs=[self.PT_total],
                    device=self.device,
                )
                launch(
                    energy.reduce_energy,
                    dim=self.PT_pair.shape[0],
                    inputs=[self.PT_friction_energies],
                    outputs=[self.friction_total],
                    device=self.device,
                )
            if self.EE_pair.shape[0] > 0:
                launch(
                    energy.reduce_energy,
                    dim=self.EE_pair.shape[0],
                    inputs=[self.EE_barrier_energies],
                    outputs=[self.EE_total],
                    device=self.device,
                )
                launch(
                    energy.reduce_energy,
                    dim=self.EE_pair.shape[0],
                    inputs=[self.EE_friction_energies],
                    outputs=[self.friction_total],
                    device=self.device,
                )
            launch(
                energy.combine_energy,
                dim=1,
                inputs=[
                    self.elastic_total,
                    self.PT_total,
                    self.EE_total,
                    self.friction_total,
                    self.kinetic_total,
                ],
                outputs=[self.total_energy],
                device=self.device,
            )

    def evaluate(self, x_np):
        """返回 (E, grad)：总能量标量和梯度 (n, 3) float64。"""
        self.tape.reset()
        self._forward(x_np, record_tape=self.tape)
        self.tape.backward(loss=self.total_energy)
        E = float(self.total_energy.numpy()[0])
        grad = self.positions.grad.numpy().astype(np.float64)
        return E, grad

    def energy_only(self, x_np):
        """只算总能量（无反向，线搜索时用，便宜）。"""
        self.tape.reset()
        self._forward(x_np, record_tape=self.tape)
        return float(self.total_energy.numpy()[0])

    # ------------------------------------------------------------------
    # FD Hessian
    # ------------------------------------------------------------------

    def hessian_fd(self, x_np, h_scale=1e-6):
        """中心差分 Hessian，只算自由度的列（固定顶点列不需要）。

        返回 (H_ff, free_idx)：H_ff 是 (Nf, Nf) 自由子矩阵，
        free_idx 是自由自由度在全局 (3n,) 里的下标。
        差分步长 h = h_scale * (1 + ||x||inf)，坐标量级 0.1 时 h ~ 1e-7。
        动能项梯度是仿射函数，中心差分对它精确，H 会自动带上 M/dt^2。
        """
        n = self.num_vertices
        free_idx = np.where(self.free_mask)[0]
        Nf = free_idx.size
        h = h_scale * (1.0 + float(np.max(np.abs(x_np))))

        H_ff = np.zeros((Nf, Nf), dtype=np.float64)
        for ii, j in enumerate(free_idx):
            if ii % 50 == 0 and ii > 0:
                print(f"    [FD] free column {ii}/{Nf}")
            xp = x_np.copy().reshape(-1)
            xp[j] += h
            _, gp = self.evaluate(xp.reshape(n, 3))
            xm = x_np.copy().reshape(-1)
            xm[j] -= h
            _, gm = self.evaluate(xm.reshape(n, 3))
            H_ff[:, ii] = (gp.reshape(-1) - gm.reshape(-1))[free_idx] / (2.0 * h)

        # 数值误差会导致轻微不对称，对称化后对角加微小正则
        H_ff = 0.5 * (H_ff + H_ff.T)
        return H_ff, free_idx

    # ------------------------------------------------------------------
    # CCD 过滤（简化二分版）
    # ------------------------------------------------------------------

    def _pair_distances(self, x_np):
        """返回 (pt_d2, ee_d2) numpy 数组。"""
        self._set_positions(x_np)
        pt_d2, ee_d2, _, _ = geometry.compute_distance(
            self.positions, self.PT_pair, self.EE_pair, max(self.d_tilde, 1e-9)
        )
        return pt_d2.numpy(), ee_d2.numpy()

    def _feasible(self, x_np):
        """候选位置是否可接受：
        1) 所有四面体体积 > 0（防止反转）；
        2) 没有任何接触对真实穿透（d^2 接近 0）。

        简化版不做精确 CCD 根求解，只用二分采样保证候选位置不进 barrier
        发散区；d_tilde 以内的接近是允许的，由 barrier 势能本身处理。
        """
        # 体积
        self._set_positions(x_np)
        wp.launch(
            signed_tet_volumes_kernel,
            dim=self.num_tets,
            inputs=[self.tet_indices, self.positions],
            outputs=[self.tet_volumes],
            device=self.device,
        )
        if float(np.min(self.tet_volumes.numpy())) <= 0.0:
            return False

        # 接触对距离
        pt_d2, ee_d2 = self._pair_distances(x_np)
        if pt_d2.size > 0 and pt_d2.min() < 1e-16:
            return False
        if ee_d2.size > 0 and ee_d2.min() < 1e-16:
            return False
        return True

    def ccd_alpha_max(self, x_np, delta_x_np):
        """在 [0,1] 上二分求最大可接受步长（保体积、不穿透）。"""
        delta = np.asarray(delta_x_np).reshape(self.num_vertices, 3)
        if not self._feasible(x_np):
            return 0.0
        if self._feasible(x_np + delta):
            return 1.0

        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if self._feasible(x_np + mid * delta):
                lo = mid
            else:
                hi = mid
        return lo

    # ------------------------------------------------------------------
    # Armijo 线搜索
    # ------------------------------------------------------------------

    def armijo(self, x_np, E0, g_flat, delta_x_flat, alpha_max, c=1e-4):
        """Armijo 充分下降：E(x + a*dx) <= E(x) + c*a*g^T*dx"""
        g_dx = float(np.dot(g_flat, delta_x_flat))
        if g_dx >= 0.0:
            return 0.0  # 非下降方向（理论上不该发生）
        alpha = alpha_max
        for _ in range(50):
            E_trial = self.energy_only(x_np + alpha * delta_x_flat.reshape(-1, 3))
            if np.isfinite(E_trial) and E_trial <= E0 + c * alpha * g_dx:
                return alpha
            alpha *= 0.5
        return 0.0

    # ------------------------------------------------------------------
    # 单步 Newton 求解
    # ------------------------------------------------------------------

    def set_step(self, x_hat_np, lagged_positions, lagged_data):
        """step 级冻结量：惯性预测点 + 摩擦滞后构型 + 摩擦滞后数据。"""
        self.tape.reset()
        wp.copy(
            self.x_hat_wp,
            wp.from_numpy(
                np.ascontiguousarray(x_hat_np, dtype=np.float32),
                dtype=wp.vec3,
                device=self.device,
            ),
        )
        self.lagged_positions = lagged_positions  # wp.array，只读引用
        self.lagged_data = lagged_data

    def solve_step(
        self,
        x_start_np,
        max_iter=20,
        tol_g=1e-6,
        tol_x=1e-8,
        verbose=True,
    ):
        """对给定 x_hat 做一次完整的 Newton 求解，返回 (x, info)。

        x_start_np: 迭代起点（惯性预测点 x_hat），(n,3) float64。
        info: dict(iterations, converged, E0, Efinal, grad_norm0, grad_norm1)
        """
        x = np.asarray(x_start_np, dtype=np.float64).copy()
        free_idx = np.where(self.free_mask)[0]
        g0_norm = 1.0
        converged = False
        E0 = float("nan")

        for it in range(max_iter):
            E, g = self.evaluate(x)
            if it == 0:
                E0 = E
                g0_norm = float(np.max(np.abs(g.reshape(-1)[free_idx])))
            g_flat = g.reshape(-1)
            # 收敛度量只看自由自由度：固定顶点是反应力，永远不会归零
            g_norm = float(np.max(np.abs(g_flat[free_idx])))

            if g_norm < tol_g * (1.0 + g0_norm):
                converged = True
                if verbose:
                    print(f"    Newton iter {it}: |g|={g_norm:.3e} 收敛(梯度)")
                break

            # FD Hessian + 解线性系统（只解自由子集，固定顶点直接不动）
            H_ff, free_idx = self.hessian_fd(x)
            g_f = g_flat[free_idx]
            try:
                delta_f = np.linalg.solve(H_ff, -g_f)
            except np.linalg.LinAlgError:
                # 奇异兜底：小正则
                delta_f = np.linalg.solve(
                    H_ff + 1e-6 * np.eye(free_idx.size), -g_f
                )
            delta = np.zeros(self.num_dofs)
            delta[free_idx] = delta_f
            g_flat[~self.free_mask] = 0.0

            # 若不是下降方向（FD Hessian 负定/数值噪声），退化为最速下降
            if float(np.dot(g_flat, delta)) >= 0.0:
                delta = np.zeros(self.num_dofs)
                delta[free_idx] = -g_f
                # g_flat 已把固定位清零，这里用负梯度方向



            alpha_max = self.ccd_alpha_max(x, delta)
            if alpha_max <= 0.0:
                if verbose:
                    print(f"    Newton iter {it}: CCD 无可行步，提前结束")
                break

            alpha = self.armijo(x, E, g_flat, delta, alpha_max)
            if alpha <= 0.0:
                if verbose:
                    print(f"    Newton iter {it}: Armijo 失败，提前结束")
                break

            step_norm = float(np.max(np.abs(alpha * delta)))
            x = x + alpha * delta.reshape(-1, 3)
            if verbose:
                print(
                    f"    Newton iter {it}: E={E:.6e} |g|={g_norm:.3e} "
                    f"alpha={alpha:.3e} step={step_norm:.3e}"
                )
            if step_norm < tol_x:
                converged = True
                if verbose:
                    print(f"    Newton iter {it}: 步长 < {tol_x} 收敛")
                break

        E_final, g_final = self.evaluate(x)
        return x, {
            "iterations": it + 1,
            "converged": converged,
            "E0": E0,
            "Efinal": E_final,
            "grad_norm0": g0_norm,
            "grad_norm1": float(np.max(np.abs(g_final.reshape(-1)[free_idx]))),
        }