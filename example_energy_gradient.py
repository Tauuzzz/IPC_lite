import numpy as np
import warp as wp

import energy
import geometry


def build_surface_candidates(tet_indices, num_tets, device):
    """从四面体提取表面,并生成PT/EE拓扑候选对."""
    all_faces = wp.empty(num_tets * 4, dtype=wp.vec3i, device=device)
    wp.launch(
        kernel=geometry.extract_all_faces,
        dim=num_tets,
        inputs=[tet_indices],
        outputs=[all_faces],
        device=device,
    )

    surface_faces, surface_edges = geometry.extract_surface_faces_and_edges(
        all_faces
    )
    PT_pair = geometry.make_PT_candidates(
        surface_faces.numpy(),
        device=device,
    )
    EE_pair = geometry.make_EE_candidates(
        surface_edges.numpy(),
        device=device,
    )

    return surface_faces, surface_edges, PT_pair, EE_pair


def main():
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    # 两个互不相交的四面体,最近表面间距为0.5毫米.
    rest_positions_np = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.2, 0.2, -0.0005],
            [1.2, 0.2, -0.0005],
            [0.2, 1.2, -0.0005],
            [0.2, 0.2, -1.0005],
        ],
        dtype=np.float32,
    )
    tet_indices_np = np.array(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ],
        dtype=np.int32,
    )

    num_vertices = rest_positions_np.shape[0]
    num_tets = tet_indices_np.shape[0]

    young_modulus = 1.0e5
    poisson_ratio = 0.1
    mu = young_modulus / (2.0 * (1.0 + poisson_ratio))
    lame_lambda = (
        young_modulus
        * poisson_ratio
        / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )
    d_tilde = 1.0e-3
    kappa = 1.0e5

    print("========== 应变能,障碍能和梯度示例 ==========")
    print(f"使用设备:{device}")
    print("\n步骤1:准备两个相互靠近但不相交的四面体")
    print("顶点位置:")
    print(rest_positions_np)
    print("四面体顶点索引:")
    print(tet_indices_np)
    print(f"杨氏模量:{young_modulus:.6e}")
    print(f"泊松比:{poisson_ratio:.6f}")
    print(f"拉梅常数 mu:{mu:.6e}")
    print(f"拉梅常数 lambda:{lame_lambda:.6e}")
    print(f"障碍距离 d_tilde:{d_tilde:.6e}")
    print(f"障碍刚度 kappa:{kappa:.6e}")

    rest_positions = wp.array(
        rest_positions_np,
        dtype=wp.vec3,
        device=device,
    )
    positions = wp.array(
        rest_positions_np,
        dtype=wp.vec3,
        device=device,
        requires_grad=True,
    )
    tet_indices = wp.array(
        tet_indices_np,
        dtype=wp.vec4i,
        device=device,
    )

    Dm = wp.empty(num_tets, dtype=wp.mat33, device=device)
    Dm_inv = wp.empty(num_tets, dtype=wp.mat33, device=device)
    rest_volumes = wp.empty(num_tets, dtype=float, device=device)

    # 参考构型数据不依赖当前位置,不需要放入Tape.
    wp.launch(
        kernel=energy.compute_rest_data,
        dim=num_tets,
        inputs=[tet_indices, rest_positions],
        outputs=[Dm, Dm_inv, rest_volumes],
        device=device,
    )

    print("\n步骤2:计算每个四面体的参考数据")
    Dm_np = Dm.numpy()
    Dm_inv_np = Dm_inv.numpy()
    rest_volumes_np = rest_volumes.numpy()
    for tet_index in range(num_tets):
        print(f"四面体 {tet_index} 的 Dm:")
        print(Dm_np[tet_index])
        print(f"四面体 {tet_index} 的 Dm_inv:")
        print(Dm_inv_np[tet_index])
        print(f"四面体 {tet_index} 的参考体积:{rest_volumes_np[tet_index]:.12e}")

    surface_faces, surface_edges, PT_pair, EE_pair = build_surface_candidates(
        tet_indices,
        num_tets,
        device,
    )

    num_PT_pairs = PT_pair.shape[0]
    num_EE_pairs = EE_pair.shape[0]

    print("\n步骤3:提取表面并生成接触候选对")
    print("表面三角形:")
    print(surface_faces.numpy())
    print("表面边:")
    print(surface_edges.numpy())
    print(f"PT候选数量:{num_PT_pairs}")
    print(f"EE候选数量:{num_EE_pairs}")

    Ds = wp.empty(num_tets, dtype=wp.mat33, device=device)
    F = wp.empty(num_tets, dtype=wp.mat33, device=device)

    # 这些数组位于positions到total_energy的计算链中,需要梯度缓冲区.
    tet_energies = wp.zeros(
        num_tets,
        dtype=float,
        device=device,
        requires_grad=True,
    )
    PT_barrier_energies = wp.zeros(
        num_PT_pairs,
        dtype=float,
        device=device,
        requires_grad=True,
    )
    EE_barrier_energies = wp.zeros(
        num_EE_pairs,
        dtype=float,
        device=device,
        requires_grad=True,
    )

    elastic_total = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    PT_total = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    EE_total = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    # 这个示例不含动能和摩擦,各占位一个全零项以匹配 combine_energy 的签名
    friction_total = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    kinetic_total = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    total_energy = wp.zeros(1, dtype=float, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel=energy.compute_spatial_data,
            dim=num_tets,
            inputs=[
                tet_indices,
                positions,
                Dm_inv,
                rest_volumes,
                mu,
                lame_lambda,
            ],
            outputs=[Ds, F, tet_energies],
            device=device,
        )

        energy.compute_barrier_energies(
            positions,
            PT_pair,
            EE_pair,
            d_tilde,
            kappa,
            PT_barrier_energies,
            EE_barrier_energies,
        )

        wp.launch(
            kernel=energy.reduce_energy,
            dim=num_tets,
            inputs=[tet_energies],
            outputs=[elastic_total],
            device=device,
        )

        if num_PT_pairs > 0:
            wp.launch(
                kernel=energy.reduce_energy,
                dim=num_PT_pairs,
                inputs=[PT_barrier_energies],
                outputs=[PT_total],
                device=device,
            )

        if num_EE_pairs > 0:
            wp.launch(
                kernel=energy.reduce_energy,
                dim=num_EE_pairs,
                inputs=[EE_barrier_energies],
                outputs=[EE_total],
                device=device,
            )

        wp.launch(
            kernel=energy.combine_energy,
            dim=1,
            inputs=[elastic_total, PT_total, EE_total, friction_total, kinetic_total],
            outputs=[total_energy],
            device=device,
        )

    print("\n步骤4:计算应变能和障碍能")
    Ds_np = Ds.numpy()
    F_np = F.numpy()
    tet_energies_np = tet_energies.numpy()
    for tet_index in range(num_tets):
        print(f"四面体 {tet_index} 的 Ds:")
        print(Ds_np[tet_index])
        print(f"四面体 {tet_index} 的 F:")
        print(F_np[tet_index])
        print(f"四面体 {tet_index} 的应变能:{tet_energies_np[tet_index]:.12e}")

    PT_pair_np = PT_pair.numpy()
    EE_pair_np = EE_pair.numpy()
    PT_barrier_energies_np = PT_barrier_energies.numpy()
    EE_barrier_energies_np = EE_barrier_energies.numpy()

    print("非零PT障碍能:")
    for pair_index in np.flatnonzero(PT_barrier_energies_np > 0.0):
        print(
            f"  候选 {pair_index},顶点索引 {PT_pair_np[pair_index]},"
            f"能量 {PT_barrier_energies_np[pair_index]:.12e}"
        )

    print("非零EE障碍能:")
    for pair_index in np.flatnonzero(EE_barrier_energies_np > 0.0):
        print(
            f"  候选 {pair_index},顶点索引 {EE_pair_np[pair_index]},"
            f"能量 {EE_barrier_energies_np[pair_index]:.12e}"
        )

    print(f"应变能总和:{float(elastic_total.numpy()[0]):.12e}")
    print(f"PT障碍能总和:{float(PT_total.numpy()[0]):.12e}")
    print(f"EE障碍能总和:{float(EE_total.numpy()[0]):.12e}")
    print(f"总能量:{float(total_energy.numpy()[0]):.12e}")

    print("\n步骤5:反向传播,计算总能量对顶点位置的梯度")
    tape.backward(loss=total_energy)

    gradient_np = positions.grad.numpy()
    print("总能量对各顶点位置的梯度:")
    print(gradient_np)


if __name__ == "__main__":
    main()
