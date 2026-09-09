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
        difference = distance_squared - d_tilde_squared
        distance_ratio = distance_squared / d_tilde_squared
        energy = (
            -kappa
            * difference
            * difference
            * wp.log(distance_ratio)
        )

    return energy


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
