import numpy as np
import warp as wp


def compute_lumped_masses(
    tet_indices_np: np.ndarray,
    rest_volumes_np: np.ndarray,
    num_vertices: int,
    rho: float = 1000.0,
) -> np.ndarray:
    """按顶点累加 lumped 质量：每个 tet 的质量 rho*vol 均分给 4 个顶点。

    Arguments:
        tet_indices_np: (num_tets, 4) int32 数组
        rest_volumes_np: (num_tets,) float 数组（参考体积）
        num_vertices: 顶点个数
        rho: 密度
    """
    masses = np.zeros(num_vertices, dtype=np.float64)
    for tet, vol in zip(tet_indices_np, rest_volumes_np):
        m = rho * vol / 4.0
        masses[tet] += m
    return masses.astype(np.float32)


@wp.kernel
def compute_kinetic_energy(
    positions: wp.array[wp.vec3],
    position_hat: wp.array[wp.vec3],
    masses: wp.array[float],
    dt: float,
    kinetic_total: wp.array[float],
):
    """动能项 K = sum_i 0.5 * m_i * ||x_i - x_hat_i||^2 / dt^2

    对应增量势能里的惯性惩罚项。必须在 wp.Tape 内调用，
    这样反向时能自动得到 M/dt^2 * (x - x_hat) 这一项梯度。
    """
    i = wp.tid()
    diff = positions[i] - position_hat[i]
    kinetic = 0.5 * masses[i] * wp.dot(diff, diff) / (dt * dt)
    wp.atomic_add(kinetic_total, 0, kinetic)