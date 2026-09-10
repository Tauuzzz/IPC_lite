from dataclasses import dataclass

import warp as wp

import geometry
from barrier_energy import barrier_force_magnitude


@dataclass
class FrictionLaggedData:
    # 一次 Newton 求解内全部冻结的摩擦几何量，都在滞后构型 x_hat 上算好
    # beta / alpha 决定相对位移算子 Gamma，tangent 决定切向基 T，normal_force 就是 lambda
    PT_beta: wp.array
    PT_tangent0: wp.array
    PT_tangent1: wp.array
    PT_normal_force: wp.array
    EE_alpha: wp.array
    EE_tangent0: wp.array
    EE_tangent1: wp.array
    EE_normal_force: wp.array


@wp.kernel
def compute_PT_friction_lagged_data_kernel(
    positions_lagged: wp.array[wp.vec3],
    PT_pair: wp.array[wp.vec4i],
    d_tilde: float,
    kappa: float,
    PT_beta: wp.array[wp.vec2],
    PT_tangent0: wp.array[wp.vec3],
    PT_tangent1: wp.array[wp.vec3],
    PT_normal_force: wp.array[float],
):
    pair_index = wp.tid()
    pair = PT_pair[pair_index]

    point = positions_lagged[pair[0]]
    A = positions_lagged[pair[1]]
    B = positions_lagged[pair[2]]
    C = positions_lagged[pair[3]]

    PT_beta[pair_index] = geometry.point_triangle_closest_point(point, A, B, C)

    tangent0, tangent1 = geometry.point_triangle_tangent_basis(A, B, C)
    PT_tangent0[pair_index] = tangent0
    PT_tangent1[pair_index] = tangent1

    distance_squared = geometry.compute_point_to_triangle_distance_squared(
        point,
        A,
        B,
        C,
    )
    PT_normal_force[pair_index] = barrier_force_magnitude(
        distance_squared,
        d_tilde,
        kappa,
    )


@wp.kernel
def compute_EE_friction_lagged_data_kernel(
    positions_lagged: wp.array[wp.vec3],
    EE_pair: wp.array[wp.vec4i],
    d_tilde: float,
    kappa: float,
    EE_alpha: wp.array[wp.vec2],
    EE_tangent0: wp.array[wp.vec3],
    EE_tangent1: wp.array[wp.vec3],
    EE_normal_force: wp.array[float],
):
    pair_index = wp.tid()
    pair = EE_pair[pair_index]

    A = positions_lagged[pair[0]]
    B = positions_lagged[pair[1]]
    C = positions_lagged[pair[2]]
    D = positions_lagged[pair[3]]

    EE_alpha[pair_index] = geometry.edge_edge_closest_point(A, B, C, D)

    tangent0, tangent1 = geometry.edge_edge_tangent_basis(A, B, C, D)
    EE_tangent0[pair_index] = tangent0
    EE_tangent1[pair_index] = tangent1

    distance_squared = geometry.compute_edge_to_edge_distance_squared(A, B, C, D)
    EE_normal_force[pair_index] = barrier_force_magnitude(
        distance_squared,
        d_tilde,
        kappa,
    )


def compute_friction_lagged_data(
    positions_lagged,
    PT_pair,
    EE_pair,
    d_tilde,
    kappa,
):
    """在滞后构型 x_hat 上算出摩擦需要的全部冻结量。

    这个函数必须在 wp.Tape 外面调用，输出数组也都不带梯度。
    """

    device = positions_lagged.device
    num_PT_pairs = PT_pair.shape[0]
    num_EE_pairs = EE_pair.shape[0]

    PT_beta = wp.empty(num_PT_pairs, dtype=wp.vec2, device=device)
    PT_tangent0 = wp.empty(num_PT_pairs, dtype=wp.vec3, device=device)
    PT_tangent1 = wp.empty(num_PT_pairs, dtype=wp.vec3, device=device)
    PT_normal_force = wp.empty(num_PT_pairs, dtype=float, device=device)

    EE_alpha = wp.empty(num_EE_pairs, dtype=wp.vec2, device=device)
    EE_tangent0 = wp.empty(num_EE_pairs, dtype=wp.vec3, device=device)
    EE_tangent1 = wp.empty(num_EE_pairs, dtype=wp.vec3, device=device)
    EE_normal_force = wp.empty(num_EE_pairs, dtype=float, device=device)

    if num_PT_pairs > 0:
        wp.launch(
            kernel=compute_PT_friction_lagged_data_kernel,
            dim=num_PT_pairs,
            inputs=[
                positions_lagged,
                PT_pair,
                d_tilde,
                kappa,
            ],
            outputs=[
                PT_beta,
                PT_tangent0,
                PT_tangent1,
                PT_normal_force,
            ],
            device=device,
        )

    if num_EE_pairs > 0:
        wp.launch(
            kernel=compute_EE_friction_lagged_data_kernel,
            dim=num_EE_pairs,
            inputs=[
                positions_lagged,
                EE_pair,
                d_tilde,
                kappa,
            ],
            outputs=[
                EE_alpha,
                EE_tangent0,
                EE_tangent1,
                EE_normal_force,
            ],
            device=device,
        )

    return FrictionLaggedData(
        PT_beta=PT_beta,
        PT_tangent0=PT_tangent0,
        PT_tangent1=PT_tangent1,
        PT_normal_force=PT_normal_force,
        EE_alpha=EE_alpha,
        EE_tangent0=EE_tangent0,
        EE_tangent1=EE_tangent1,
        EE_normal_force=EE_normal_force,
    )


@wp.func
def friction_f0(
    slip_norm: float,
    y_eps: float,
) -> float:
    # 库仑摩擦 |y| 的 C1 光滑化
    # slip_norm >= y_eps：纯滑动，就是 |y| 本身
    # slip_norm <  y_eps：三次多项式，把 |y| 在 0 处的尖角抹平（静摩擦区）
    # 两段在 y_eps 处值和导数都接得上：f0(y_eps) = y_eps，f0'(y_eps) = 1
    if slip_norm >= y_eps:
        return slip_norm

    return (
        slip_norm * slip_norm / y_eps
        - slip_norm * slip_norm * slip_norm / (3.0 * y_eps * y_eps)
        + y_eps / 3.0
    )


@wp.kernel
def compute_PT_friction_energy_kernel(
    positions: wp.array[wp.vec3],
    positions_lagged: wp.array[wp.vec3],
    PT_pair: wp.array[wp.vec4i],
    PT_beta: wp.array[wp.vec2],
    PT_tangent0: wp.array[wp.vec3],
    PT_tangent1: wp.array[wp.vec3],
    PT_normal_force: wp.array[float],
    mu_friction: float,
    y_eps: float,
    PT_friction_energies: wp.array[float],
):
    pair_index = wp.tid()
    pair = PT_pair[pair_index]

    beta = PT_beta[pair_index]
    beta1 = beta[0]
    beta2 = beta[1]

    # 各顶点从滞后构型 x_hat 到当前构型 x 的位移
    delta_point = positions[pair[0]] - positions_lagged[pair[0]]
    delta_A = positions[pair[1]] - positions_lagged[pair[1]]
    delta_B = positions[pair[2]] - positions_lagged[pair[2]]
    delta_C = positions[pair[3]] - positions_lagged[pair[3]]

    # 接触点上的相对位移 Gamma @ delta_x
    # 三角形上那个材料点的位移是三个顶点位移的重心混合
    relative_displacement = (
        delta_point
        + (beta1 + beta2 - 1.0) * delta_A
        - beta1 * delta_B
        - beta2 * delta_C
    )

    # 投影到切平面，得到滑动位移 y；沿法线的分量被丢掉，那部分归 barrier 管
    slip = wp.vec2(
        wp.dot(PT_tangent0[pair_index], relative_displacement),
        wp.dot(PT_tangent1[pair_index], relative_displacement),
    )
    # 这里必须用 wp.length：slip = 0 时它的梯度是 0，
    # 而 wp.sqrt(wp.dot(slip, slip)) 的梯度是 NaN
    slip_norm = wp.length(slip)

    PT_friction_energies[pair_index] = (
        mu_friction
        * PT_normal_force[pair_index]
        * friction_f0(slip_norm, y_eps)
    )


@wp.kernel
def compute_EE_friction_energy_kernel(
    positions: wp.array[wp.vec3],
    positions_lagged: wp.array[wp.vec3],
    EE_pair: wp.array[wp.vec4i],
    EE_alpha: wp.array[wp.vec2],
    EE_tangent0: wp.array[wp.vec3],
    EE_tangent1: wp.array[wp.vec3],
    EE_normal_force: wp.array[float],
    mu_friction: float,
    y_eps: float,
    EE_friction_energies: wp.array[float],
):
    pair_index = wp.tid()
    pair = EE_pair[pair_index]

    alpha = EE_alpha[pair_index]
    alpha1 = alpha[0]
    alpha2 = alpha[1]

    delta_A = positions[pair[0]] - positions_lagged[pair[0]]
    delta_B = positions[pair[1]] - positions_lagged[pair[1]]
    delta_C = positions[pair[2]] - positions_lagged[pair[2]]
    delta_D = positions[pair[3]] - positions_lagged[pair[3]]

    # 两条边上各自那个材料点的位移之差
    relative_displacement = (
        (1.0 - alpha1) * delta_A
        + alpha1 * delta_B
        - (1.0 - alpha2) * delta_C
        - alpha2 * delta_D
    )
    # 投影到切平面，得到滑动位移 y；沿法线的分量被丢掉，那部分归 barrier 管
    slip = wp.vec2(
        wp.dot(EE_tangent0[pair_index], relative_displacement),
        wp.dot(EE_tangent1[pair_index], relative_displacement),
    )
    slip_norm = wp.length(slip)

    EE_friction_energies[pair_index] = (
        mu_friction
        * EE_normal_force[pair_index]
        * friction_f0(slip_norm, y_eps)
    )


def compute_friction_energies(
    positions,
    positions_lagged,
    PT_pair,
    EE_pair,
    lagged_data,
    mu_friction,
    y_eps,
    PT_friction_energies,
    EE_friction_energies,
):
    """计算全部 PT/EE 候选对的摩擦耗散能。

    必须在 wp.Tape 里面调用。lagged_data 里的量全部是冻结常量，不带梯度。
    """

    device = positions.device
    num_PT_pairs = PT_pair.shape[0]
    num_EE_pairs = EE_pair.shape[0]

    if num_PT_pairs > 0:
        wp.launch(
            kernel=compute_PT_friction_energy_kernel,
            dim=num_PT_pairs,
            inputs=[
                positions,
                positions_lagged,
                PT_pair,
                lagged_data.PT_beta,
                lagged_data.PT_tangent0,
                lagged_data.PT_tangent1,
                lagged_data.PT_normal_force,
                mu_friction,
                y_eps,
            ],
            outputs=[PT_friction_energies],
            device=device,
        )

    if num_EE_pairs > 0:
        wp.launch(
            kernel=compute_EE_friction_energy_kernel,
            dim=num_EE_pairs,
            inputs=[
                positions,
                positions_lagged,
                EE_pair,
                lagged_data.EE_alpha,
                lagged_data.EE_tangent0,
                lagged_data.EE_tangent1,
                lagged_data.EE_normal_force,
                mu_friction,
                y_eps,
            ],
            outputs=[EE_friction_energies],
            device=device,
        )

    return PT_friction_energies, EE_friction_energies