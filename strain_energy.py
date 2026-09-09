import warp as wp


@wp.kernel
def compute_rest_data(
    tet_indices: wp.array[wp.vec4i],
    rest_positions: wp.array[wp.vec3],
    Dm: wp.array[wp.mat33],
    Dm_inv: wp.array[wp.mat33],
    rest_volumes: wp.array[float],
):
    tet_index = wp.tid()
    tet = tet_indices[tet_index]

    i0 = tet[0]
    i1 = tet[1]
    i2 = tet[2]
    i3 = tet[3]

    x0 = rest_positions[i0]
    x1 = rest_positions[i1]
    x2 = rest_positions[i2]
    x3 = rest_positions[i3]

    Dm[tet_index] = wp.matrix_from_cols(x1 - x0, x2 - x0, x3 - x0)
    Dm_inv[tet_index] = wp.inverse(Dm[tet_index])
    rest_volumes[tet_index] = wp.abs(wp.determinant(Dm[tet_index])) / 6.0


@wp.kernel
def compute_spatial_data(
    tet_indices: wp.array[wp.vec4i],
    positions: wp.array[wp.vec3],
    Dm_inv: wp.array[wp.mat33],
    rest_volumes: wp.array[float],
    mu: float,
    lamda: float,
    Ds: wp.array[wp.mat33],
    F: wp.array[wp.mat33],
    tet_energies: wp.array[float],
):
    tet_index = wp.tid()
    tet = tet_indices[tet_index]

    i0 = tet[0]
    i1 = tet[1]
    i2 = tet[2]
    i3 = tet[3]

    x0 = positions[i0]
    x1 = positions[i1]
    x2 = positions[i2]
    x3 = positions[i3]

    Ds_value = wp.matrix_from_cols(x1 - x0, x2 - x0, x3 - x0)
    F_value = Ds_value @ Dm_inv[tet_index]

    Ds[tet_index] = Ds_value
    F[tet_index] = F_value

    I1 = wp.trace(F_value @ wp.transpose(F_value))
    J = wp.determinant(F_value)
    phi = mu * 0.5 * (I1 - 3.0) - mu * wp.log(J) + 0.5 * lamda * wp.log(J) * wp.log(J)
    tet_energies[tet_index] = phi * rest_volumes[tet_index]
