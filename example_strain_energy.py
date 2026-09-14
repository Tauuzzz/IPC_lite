import numpy as np
import warp as wp

import energy


def run_example(
    case_name,
    positions_np,
    tet_indices,
    rest_inv,
    rest_volumes,
    mu,
    lame_lambda,
    device,
):
    """调用energy.py中的kernel,计算并输出一个构型的应变能."""
    num_tets = len(tet_indices)

    # 把当前位置传到 Warp 设备上.
    positions = wp.array(positions_np, dtype=wp.vec3, device=device)

    # 为energy.py中kernel的三个输出分配空间.
    spatial_edge_matrices = wp.empty(num_tets, dtype=wp.mat33, device=device)
    deformation_gradients = wp.empty(num_tets, dtype=wp.mat33, device=device)
    tet_energies = wp.empty(num_tets, dtype=float, device=device)

    # 调用energy.py中计算四面体应变能的kernel.
    wp.launch(
        kernel=energy.compute_spatial_data,
        dim=num_tets,
        inputs=[
            tet_indices,
            positions,
            rest_inv,
            rest_volumes,
            mu,
            lame_lambda,
        ],
        outputs=[
            spatial_edge_matrices,
            deformation_gradients,
            tet_energies,
        ],
        device=device,
    )

    F_np = deformation_gradients.numpy()[0]
    tet_energy = float(tet_energies.numpy()[0])
    rest_volume = float(rest_volumes.numpy()[0])

    print(f"\n案例:{case_name}")
    print("当前边矩阵 Ds:")
    print(spatial_edge_matrices.numpy()[0])
    print("形变梯度 F:")
    print(F_np)
    print(f"体积比 J:{float(np.linalg.det(F_np)):.6f}")
    print(f"应变能密度 phi:{tet_energy / rest_volume:.6f}")
    print(f"四面体应变能:{tet_energy:.6f}")


def main():
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    # 材料参数.
    young_modulus = 1.0e5
    poisson_ratio = 0.1
    mu = young_modulus / (2.0 * (1.0 + poisson_ratio))
    lame_lambda = (
        young_modulus
        * poisson_ratio
        / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )

    # 一个单位四面体,参考边矩阵 Dm 是单位矩阵.
    rest_positions_np = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    tet_indices_np = np.array([[0, 1, 2, 3]], dtype=np.int32)

    rest_positions = wp.array(rest_positions_np, dtype=wp.vec3, device=device)
    tet_indices = wp.array(tet_indices_np, dtype=wp.vec4i, device=device)
    Dm = wp.empty(1, dtype=wp.mat33, device=device)
    rest_inv = wp.empty(1, dtype=wp.mat33, device=device)
    rest_volumes = wp.empty(1, dtype=float, device=device)

    # 调用energy.py中的参考构型预计算kernel.
    wp.launch(
        kernel=energy.compute_rest_data,
        dim=1,
        inputs=[tet_indices, rest_positions],
        outputs=[Dm, rest_inv, rest_volumes],
        device=device,
    )

    print(f"使用设备:{device}")
    print("参考边矩阵 Dm:")
    print(Dm.numpy()[0])
    print(f"参考体积:{float(rest_volumes.numpy()[0]):.6f}")

    # 未发生形变时,F 是单位矩阵,应变能应为零.
    run_example(
        "未发生形变",
        rest_positions_np,
        tet_indices,
        rest_inv,
        rest_volumes,
        mu,
        lame_lambda,
        device,
    )

    # 将顶点 1 沿 x 方向拉伸 10%.
    stretched_positions_np = rest_positions_np.copy()
    stretched_positions_np[1, 0] = 1.1
    run_example(
        "沿 x 方向拉伸 10%",
        stretched_positions_np,
        tet_indices,
        rest_inv,
        rest_volumes,
        mu,
        lame_lambda,
        device,
    )


if __name__ == "__main__":
    main()
