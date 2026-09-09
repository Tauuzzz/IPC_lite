import warp as wp

@wp.kernel
def compute_friction_lagged_data(positions_lagged, PT_pair, EE_pair, d_tilde, kappa):
    index = wp.tid()

    if index < PT_pair.shape[0]:
        i = PT_pair[index, 0]
        j = PT_pair[index, 1]

        # Compute the friction energy for the PT pair
        friction_energy = kappa * wp.length(positions_lagged[i] - positions_lagged[j]) / d_tilde
        friction_energies[index] = friction_energy

    elif index < PT_pair.shape[0] + EE_pair.shape[0]:
        ee_index = index - PT_pair.shape[0]
        i = EE_pair[ee_index, 0]
        j = EE_pair[ee_index, 1]

        # Compute the friction energy for the EE pair
        friction_energy = kappa * wp.length(positions_lagged[i] - positions_lagged[j]) / d_tilde
        friction_energies[index] = friction_energy

@wp.kernel
def compute_friction_energy(positions, positions_lagged, PT_pair, EE_pair, d_tilde, kappa, friction_energies):
    index = wp.tid()

    if index < PT_pair.shape[0]:
        i = PT_pair[index, 0]
        j = PT_pair[index, 1]

        # Compute the friction energy for the PT pair
        friction_energy = kappa * wp.length(positions[i] - positions_lagged[j]) / d_tilde
        friction_energies[index] = friction_energy

    elif index < PT_pair.shape[0] + EE_pair.shape[0]:
        ee_index = index - PT_pair.shape[0]
        i = EE_pair[ee_index, 0]
        j = EE_pair[ee_index, 1]

        # Compute the friction energy for the EE pair
        friction_energy = kappa * wp.length(positions[i] - positions_lagged[j]) / d_tilde
        friction_energies[index] = friction_energy
