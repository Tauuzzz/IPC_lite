"""IPC Newton 求解器(教学版,不为性能优化).

这个模块负责"单个时间步内 Newton 迭代"所需的一切底层机制,
对外只暴露与伪代码编号一一对应的方法:

    伪代码(迭代内) 1~5 : 能量求值(动能/弹性/barrier/摩擦/求和)
    伪代码(迭代内) 6   : evaluate_energy_and_gradient()(梯度)
                         compute_hessian_fd()(二阶梯度,中心差分)
    伪代码(迭代内) 7   : compute_newton_direction()
    伪代码(迭代内) 8   : ccd_alpha_max()
    伪代码(迭代内) 9   : armijo_line_search()
    伪代码(迭代内) 11  : 收敛判断由 main.py 完成(逻辑简单,放主流程可见)

核心设计(教学强调):
- 所有 Warp kernel 只接受 wp.array;numpy 数组无法求梯度.
  因此"两个位置系统":numpy 是状态(float64,存 Python 侧),
  wp.array 是 GPU 计算暂存(float32,带 requires_grad).
- tape 只支持一阶反向自动微分:梯度自动算,Hessian 用中心差分
  H[:, dof_index] = (g(x+fd_step·e_j) − g(x−fd_step·e_j)) / (2h),每个自由度 2 次梯度求值.
- 摩擦 lagged 数据(β/α,切平面,法向力 λ)在 step 开头冻结,
  Newton 期间只读不写 ---- 目标函数固定,Newton 才有收敛性.
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
    """每个四面体的带符号体积(CCD 反转检查用)."""
    tet_index = wp.tid()
    tet = tet_indices[tet_index]
    x0 = positions[tet[0]]
    x1 = positions[tet[1]]
    x2 = positions[tet[2]]
    x3 = positions[tet[3]]
    D = wp.matrix_from_cols(x1 - x0, x2 - x0, x3 - x0)
    volumes[tet_index] = wp.determinant(D) / 6.0


class NewtonSolver:
    """持有全部 Warp 能量数组与 tape,提供 Newton 迭代所需的原语."""

    def __init__(
        self,
        device,
        num_vertices,
        num_tets,
        positions_wp,       # wp.array vec3, requires_grad=True
        tet_indices,        # wp.array vec4i
        Dm_inv,             # wp.array mat33(参考边矩阵的逆)
        rest_volumes,       # wp.array float(参考体积)
        mu,
        lamda,
        PT_pair,            # wp.array vec4i
        EE_pair,            # wp.array vec4i
        d_tilde,
        kappa,
        friction_mu,
        y_eps,
        masses_np,          # numpy (num_vertices,) lumped 质量
        dt,
        pinned_dof_mask=None,   # numpy bool (3*num_vertices,):固定自由度掩码
    ):
        self.device = device
        self.num_vertices = num_vertices
        self.num_tets = num_tets

        # ---- 位置状态(三个都是 wp.array,kernel 直接读) ----
        self.positions_wp = positions_wp         # 猜测位置(迭代变量,需梯度)
        self.positions_lagged_wp = wp.empty(num_vertices, dtype=wp.vec3, device=device)  # 滞后位置 x^n
        self.positions_hat_wp = wp.empty(num_vertices, dtype=wp.vec3, device=device)     # 惯性位置 x̂

        # ---- 弹性参数与网格常量 ----
        self.tet_indices = tet_indices
        self.Dm_inv = Dm_inv
        self.rest_volumes = rest_volumes
        self.mu = mu
        self.lamda = lamda

        # ---- 接触候选对与接触参数 ----
        self.PT_pair = PT_pair
        self.EE_pair = EE_pair
        self.d_tilde = d_tilde
        self.kappa = kappa

        # ---- 摩擦参数 ----
        self.friction_mu = friction_mu
        self.y_eps = y_eps
        self.lagged_data = None   # 由 main 在 step 开头注入,Newton 期间冻结

        # ---- 质量与时间步 ----
        self.masses_np = np.asarray(masses_np, dtype=np.float64).reshape(-1)
        self.masses_wp = wp.array(self.masses_np.astype(np.float32), device=device)
        self.dt = dt

        # ---- 固定自由度 ----
        self.num_dofs = 3 * num_vertices
        if pinned_dof_mask is None:
            pinned_dof_mask = np.zeros(self.num_dofs, dtype=bool)
        self.free_mask = ~np.asarray(pinned_dof_mask, dtype=bool).reshape(-1)
        self.free_dof_indices = np.where(self.free_mask)[0]

        # ---- 能量输出数组(reduce 目标都必须 requires_grad) ----
        self.Ds = wp.empty(num_tets, dtype=wp.mat33, device=device)   # 当前边矩阵
        self.F = wp.empty(num_tets, dtype=wp.mat33, device=device)    # 变形梯度
        self.tet_energies = wp.empty(num_tets, dtype=float, device=device, requires_grad=True)
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
        # 归约目标:动能/弹性/barrier(PT)/barrier(EE)/摩擦 分别归约再合并
        self.kinetic_total = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        self.elastic_total = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        self.PT_total = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        self.EE_total = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        self.friction_total = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        self.total_energy = wp.zeros(1, dtype=float, device=device, requires_grad=True)

        self.tet_volumes = wp.empty(num_tets, dtype=float, device=device)  # CCD 用
        self.tape = wp.Tape()

    # ==================================================================
    # step 级状态注入(伪代码:step 级 2/3)
    # snapshot, step 级 1(滞后位置)由 main.py 直接 wp.copy,见 main.py 循环
    # ==================================================================

    def set_hat_position(self, positions_hat_np):
        """伪代码 step 级 2:写入惯性位置 x̂(动能项的锚点)."""
        wp.copy(
            self.positions_hat_wp,
            wp.from_numpy(
                np.ascontiguousarray(positions_hat_np, dtype=np.float32),
                dtype=wp.vec3,
                device=self.device,
            ),
        )

    def set_lagged_data(self, lagged_data):
        """伪代码 step 级 3:注入 lagged 摩擦数据(β/α,切平面,λ)."""
        self.lagged_data = lagged_data

    # ==================================================================
    # 内部工具
    # ==================================================================

    def _set_positions_wp(self, positions_np):
        """把 numpy 位置的猜测位置写入 Warp positions_wp 数组(float32)."""
        wp.copy(
            self.positions_wp,
            wp.from_numpy(
                np.ascontiguousarray(positions_np, dtype=np.float32),
                dtype=wp.vec3,
                device=self.device,
            ),
        )

    def _zero_targets(self):
        """归约目标清零.

        reduce_energy 用 atomic_add 累加,不清零会跨求值一直累加.
        """
        self.kinetic_total.zero_()
        self.elastic_total.zero_()
        self.PT_total.zero_()
        self.EE_total.zero_()
        self.friction_total.zero_()
        self.total_energy.zero_()

    def _launch_all_energies(self, positions_np, record_tape):
        """伪代码(迭代内)1~5:在猜测位置上计算全部四项能量并求和.

        evaluate / energy_only 的统一入口,保证三件事缺一不可:
        1) _set_positions_wp:把 numpy 猜测位置写入 warp positions_wp(kernel 只认它);
        2) _zero_targets:清空归约目标(reduce_energy 用 atomic_add,不清会累加);
        3) 依次 launch 四项能量并合并为 total_energy.
        """
        self._set_positions_wp(positions_np)
        self._zero_targets()
        launch = wp.launch
        with record_tape:

            # ---- 1. 动能(惯性势能):锚点是 x̂ ---- 
            launch(
                compute_kinetic_energy,
                dim=self.num_vertices,
                inputs=[self.positions_wp, self.positions_hat_wp, self.masses_wp, self.dt],
                outputs=[self.kinetic_total],
                device=self.device,
            )

            # ---- 2. 弹性能(每个四面体逐个算)----
            launch(
                energy.compute_spatial_data,
                dim=self.num_tets,
                inputs=[
                    self.tet_indices,
                    self.positions_wp,
                    self.Dm_inv,
                    self.rest_volumes,
                    self.mu,
                    self.lamda,
                ],
                outputs=[self.Ds, self.F, self.tet_energies],
                device=self.device,
            )

            # ---- 3. Barrier 接触能(每个接触对逐个算)----
            energy.compute_barrier_energies(
                self.positions_wp,
                self.PT_pair,
                self.EE_pair,
                self.d_tilde,
                self.kappa,
                self.PT_barrier_energies,
                self.EE_barrier_energies,
            )

            # ---- 4. 摩擦耗散能(lagged 数据冻结,只读)----
            energy.compute_friction_energies(
                self.positions_wp,
                self.positions_lagged_wp,
                self.PT_pair,
                self.EE_pair,
                self.lagged_data,
                self.friction_mu,
                self.y_eps,
                self.PT_friction_energies,
                self.EE_friction_energies,
            )

            # ---- 5. 分别归约 + 合并成总能量 ----
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

    # ==================================================================
    # 伪代码(迭代内)6:能量,梯度,Hessian
    # ==================================================================

    def evaluate_energy_and_gradient(self, positions_np):
        """在猜测位置 x 上算总势能 E(x) 和梯度 g = ∇E(x).

        梯度由 Warp tape 反向自动微分得到(不需要手写公式),
        tape.reset() 会顺带清零所有 .grad,不会跨次累加.
        返回 (potential_energy, gradient), gradient 是 (num_vertices, 3) float64.
        """
        self.tape.reset()
        self._launch_all_energies(positions_np, record_tape=self.tape)
        self.tape.backward(loss=self.total_energy)
        potential_energy = float(self.total_energy.numpy()[0])
        gradient = self.positions_wp.grad.numpy().astype(np.float64)
        return potential_energy, gradient

    def energy_only(self, positions_np):
        """只算总能量(不 backward,Armijo 线搜索用,更便宜)."""
        self.tape.reset()
        self._launch_all_energies(positions_np, record_tape=self.tape)
        return float(self.total_energy.numpy()[0])

    def compute_hessian_fd(self, positions_np, fd_step_scale=1e-6):
        """伪代码(迭代内)6b:二阶梯度 Hessian(中心差分,稠密).

        对每个自由自由度 dof_index,扰动 ±fd_step 各算一次梯度:
            H[:, dof_index] = (g(x + fd_step·e_j) − g(x − fd_step·e_j)) / (2h)
        返回 (hessian_free, free_dof_indices):hessian_free 是自由自由度子矩阵 (num_free_dofs, num_free_dofs).
        差分步长 fd_step = fd_step_scale·(1+‖x‖∞),坐标量级 0.1 时 fd_step ~ 1e-7.
        动能项梯度是仿射函数,中心差分对它精确,H 自动含 M/Δt².
        """
        num_vertices = self.num_vertices
        free_dof_indices = self.free_dof_indices
        num_free_dofs = free_dof_indices.size
        fd_step = fd_step_scale * (1.0 + float(np.max(np.abs(positions_np))))

        hessian_free = np.zeros((num_free_dofs, num_free_dofs), dtype=np.float64)
        for column_index, dof_index in enumerate(free_dof_indices):
            if column_index % 50 == 0 and column_index > 0:
                print(f"    [FD] 自由列 {column_index}/{num_free_dofs}")
            positions_plus = positions_np.copy().reshape(-1)
            positions_plus[dof_index] += fd_step
            _, gradient_plus = self.evaluate_energy_and_gradient(positions_plus.reshape(num_vertices, 3))
            positions_minus = positions_np.copy().reshape(-1)
            positions_minus[dof_index] -= fd_step
            _, gradient_minus = self.evaluate_energy_and_gradient(positions_minus.reshape(num_vertices, 3))
            hessian_free[:, column_index] = (gradient_plus.reshape(-1) - gradient_minus.reshape(-1))[free_dof_indices] / (2.0 * fd_step)

        # 数值误差会引入轻微不对称,对称化处理
        hessian_free = 0.5 * (hessian_free + hessian_free.T)
        return hessian_free, free_dof_indices

    # ==================================================================
    # 伪代码(迭代内)7:Newton 方向
    # ==================================================================

    def compute_newton_direction(self, hessian_free, gradient_np, free_dof_indices):
        """解线性系统 (H + M/Δt²)·direction = −g,得到 Newton 方向 direction.

        说明:
        - hessian_free 只含自由自由度;固定自由度 direction=0(不动).
        - 动能项的贡献 M/Δt² 已隐含在 FD Hessian 里(差分对仿射项精确).
        - 若方向不是下降方向(g·direction ≥ 0,Hessian 数值噪声可能造成),
          退化为最速下降方向 −g(保证 Armijo 有条件执行).
        """
        gradient_free = gradient_np.reshape(-1)[free_dof_indices]
        try:
            direction_free = np.linalg.solve(hessian_free, -gradient_free)
        except np.linalg.LinAlgError:
            # 奇异兜底:加小正则
            direction_free = np.linalg.solve(hessian_free + 1e-6 * np.eye(free_dof_indices.size), -gradient_free)

        direction = np.zeros(self.num_dofs)
        direction[free_dof_indices] = direction_free

        gradient_flat = gradient_np.reshape(-1)
        if float(np.dot(gradient_flat, direction)) >= 0.0:      # 非下降方向
            direction = np.zeros(self.num_dofs)
            direction[free_dof_indices] = -gradient_free
        return direction

    # ==================================================================
    # 伪代码(迭代内)8:CCD 最大允许步长
    # ==================================================================

    def _pair_distances(self, positions_np):
        """返回 (PT 各对 d², EE 各对 d²) numpy 数组."""
        self._set_positions_wp(positions_np)
        pt_distances_squared, ee_distances_squared, _, _ = geometry.compute_distance(
            self.positions_wp, self.PT_pair, self.EE_pair, max(self.d_tilde, 1e-9)
        )
        return pt_distances_squared.numpy(), ee_distances_squared.numpy()

    def _feasible(self, positions_np):
        """候选位置是否可接受(安全条件,两个):
        1) 所有四面体体积 > 0 ---- 不允许反转;
        2) 所有接触对 d² 保持 > 0 ---- 不允许穿透(只防真实穿越,
           d < d_tilde 的接近是允许的,由 barrier 势能自己处理).
        """
        # ① 体积检查
        self._set_positions_wp(positions_np)
        wp.launch(
            signed_tet_volumes_kernel,
            dim=self.num_tets,
            inputs=[self.tet_indices, self.positions_wp],
            outputs=[self.tet_volumes],
            device=self.device,
        )
        if float(np.min(self.tet_volumes.numpy())) <= 0.0:
            return False

        # ② 穿透检查
        pt_distances_squared, ee_distances_squared = self._pair_distances(positions_np)
        if pt_distances_squared.size > 0 and pt_distances_squared.min() < 1e-16:
            return False
        if ee_distances_squared.size > 0 and ee_distances_squared.min() < 1e-16:
            return False
        return True

    def ccd_alpha_max(self, positions_np, direction_np):
        """伪代码(迭代内)8:在 [0,1] 上二分求最大安全步长 α_max.

        连续碰撞检测的简化实现:不精确求每个接触对的穿越时刻,
        而是对 α 做二分采样,每个采样点做"体积+穿透"可行性检查.
        靠"可行→不可行"的单调性收敛到边界(偏保守,安全方向).
        """
        direction = np.asarray(direction_np).reshape(self.num_vertices, 3)
        if not self._feasible(positions_np):          # 起点就不可行:一步都不能走
            return 0.0
        if self._feasible(positions_np + direction):  # 全步可行:直接 α_max = 1
            return 1.0

        alpha_lower, alpha_upper = 0.0, 1.0
        for _ in range(30):
            alpha_mid = 0.5 * (alpha_lower + alpha_upper)
            if self._feasible(positions_np + alpha_mid * direction):
                alpha_lower = alpha_mid                      # 可行 → 还能多走
            else:
                alpha_upper = alpha_mid                      # 不可行 → 回退
        return alpha_lower

    # ==================================================================
    # 伪代码(迭代内)9:Armijo 线搜索
    # ==================================================================

    def armijo_line_search(self, positions_np, energy_at_x, gradient_np, direction_np, alpha_max, sufficient_decrease=1e-4):
        """伪代码(迭代内)9:从 α_max 往回退,找满足充分下降的 α.

        Armijo 条件:E(x + α·p) ≤ E(x) + c·α·gᵀ·p (E 即 potential_energy, c 即 sufficient_decrease)
        即"这一步带来的能量下降至少要达到 斜率·步长 的 sufficient_decrease 倍".
        sufficient_decrease 一般取 1e-4(宽松,主要是排除几乎不下降的步子).
        """
        gradient_dot_direction = float(np.dot(gradient_np.reshape(-1), direction_np))
        if gradient_dot_direction >= 0.0:
            return 0.0                        # 非下降方向(理论不该发生)

        alpha = alpha_max
        for _ in range(50):
            energy_trial = self.energy_only(positions_np + alpha * direction_np.reshape(-1, 3))
            if np.isfinite(energy_trial) and energy_trial <= energy_at_x + sufficient_decrease * alpha * gradient_dot_direction:
                return alpha
            alpha *= 0.5                      # 不满足 → 步长减半重试
        return 0.0