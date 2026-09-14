import warp as wp

import geometry


@wp.func
def barrier_energy(
    distance_squared: float,
    d_tilde: float,
    kappa: float,
) -> float:
    d_tilde_squared = d_tilde * d_tilde
    energy = 0.0

    if distance_squared < d_tilde_squared:
        # 安全钳:特征恰好重合(顶点落在面/边上)时 d^2 可能精确为 0,
        # log(0) 会让能量发散且梯度变 nan.钳到极小正值,力是有限大但
        # 巨大,Newton 能正常把物体推出重叠区.正常接触路径 d 只会
        # 停在 d_tilde ~ 1e-3,永远不会到这量级,纯属保险.
        if distance_squared <= 1.0e-12:
            distance_squared = 1.0e-12
        difference = distance_squared - d_tilde_squared
        distance_ratio = distance_squared / d_tilde_squared
        energy = (
            -kappa
            * difference
            * difference
            * wp.log(distance_ratio)
        )

    return energy


@wp.func
def barrier_force_magnitude(
    distance_squared: float,
    d_tilde: float,
    kappa: float,
) -> float:
    # 法向接触力大小 N = -db/dd,也就是摩擦公式里的 lambda
    # 非激活 (d >= d_tilde) 时返回 0,摩擦能随之自动为 0
    d_tilde_squared = d_tilde * d_tilde

    if distance_squared >= d_tilde_squared:
        return 0.0

    # d = 0 说明已经穿透,交给 CCD 去避免,这里只做保护
    if distance_squared <= 1.0e-24:
        return 0.0

    difference = distance_squared - d_tilde_squared
    ratio = distance_squared / d_tilde_squared

    # b(u) = -kappa * (u - u_tilde)^2 * log(u / u_tilde),u = d^2
    # db/du = -kappa * [ 2(u - u_tilde) * log(u/u_tilde) + (u - u_tilde)^2 / u ]
    db_du = -kappa * (
        2.0 * difference * wp.log(ratio)
        + difference * difference / distance_squared
    )

    # N = -db/dd = -db/du * du/dd = -db/du * 2d
    return -db_du * 2.0 * wp.sqrt(distance_squared)


@wp.kernel
def compute_PT_barrier_energy_kernel(
    positions: wp.array[wp.vec3],
    PT_pair: wp.array[wp.vec4i],
    d_tilde: float,
    kappa: float,
    PT_barrier_energies: wp.array[float],
):
    pair_index = wp.tid()
    pair = PT_pair[pair_index]

    point = positions[pair[0]]
    A = positions[pair[1]]
    B = positions[pair[2]]
    C = positions[pair[3]]

    distance_squared = geometry.compute_point_to_triangle_distance_squared(
        point,
        A,
        B,
        C,
    )
    PT_barrier_energies[pair_index] = barrier_energy(
        distance_squared,
        d_tilde,
        kappa,
    )


@wp.kernel
def compute_EE_barrier_energy_kernel(
    positions: wp.array[wp.vec3],
    EE_pair: wp.array[wp.vec4i],
    d_tilde: float,
    kappa: float,
    EE_barrier_energies: wp.array[float],
):
    pair_index = wp.tid()
    pair = EE_pair[pair_index]

    A = positions[pair[0]]
    B = positions[pair[1]]
    C = positions[pair[2]]
    D = positions[pair[3]]

    distance_squared = geometry.compute_edge_to_edge_distance_squared(
        A,
        B,
        C,
        D,
    )
    EE_barrier_energies[pair_index] = barrier_energy(
        distance_squared,
        d_tilde,
        kappa,
    )


def compute_barrier_energies(
    positions,
    PT_pair,
    EE_pair,
    d_tilde,
    kappa,
    PT_barrier_energies,
    EE_barrier_energies,
):
    device = positions.device
    num_PT_pairs = PT_pair.shape[0]
    num_EE_pairs = EE_pair.shape[0]

    if num_PT_pairs > 0:
        wp.launch(
            kernel=compute_PT_barrier_energy_kernel,
            dim=num_PT_pairs,
            inputs=[
                positions,
                PT_pair,
                d_tilde,
                kappa,
            ],
            outputs=[PT_barrier_energies],
            device=device,
        )

    if num_EE_pairs > 0:
        wp.launch(
            kernel=compute_EE_barrier_energy_kernel,
            dim=num_EE_pairs,
            inputs=[
                positions,
                EE_pair,
                d_tilde,
                kappa,
            ],
            outputs=[EE_barrier_energies],
            device=device,
        )

    return PT_barrier_energies, EE_barrier_energies
