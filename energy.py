import warp as wp

from strain_energy import compute_rest_data, compute_spatial_data
from barrier_energy import (
    barrier_energy,
    compute_PT_barrier_energy_kernel,
    compute_EE_barrier_energy_kernel,
    compute_barrier_energies,
)
from friction_energy import (
    friction_f0,
    FrictionLaggedData,
    compute_friction_lagged_data,
    compute_friction_energies,
)


@wp.kernel
def reduce_energy(
    energies: wp.array[float],
    total: wp.array[float],
):
    index = wp.tid()
    wp.atomic_add(total, 0, energies[index])


@wp.kernel
def combine_energy(
    elastic_total: wp.array[float],
    PT_total: wp.array[float],
    EE_total: wp.array[float],
    friction_total: wp.array[float],
    kinetic_total: wp.array[float],
    total_energy: wp.array[float],
):
    total_energy[0] = (
        elastic_total[0]
        + PT_total[0]
        + EE_total[0]
        + friction_total[0]
        + kinetic_total[0]
    )
