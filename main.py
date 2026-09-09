import warp as wp
import numpy as np

import geometry
import energy

device="cuda:0"

num_vertices = 1000
num_tets=500

# Strain energy constants
NIU=0.1
E=1e5

# Lame constants
mu=E/(2*(1+NIU))
lamda=E*NIU/((1+NIU)*(1-2*NIU))

# 参考位置
rest_positions=wp.empty(num_vertices, dtype=wp.vec3, device=device)
# 当前位置
positions=wp.empty(num_vertices, dtype=wp.vec3, device=device, requires_grad=True)
# 四面体的顶点索引序号
tet_indices=wp.empty(num_tets, dtype=wp.vec4i, device=device)
# 四面体参考体积
rest_volumes=wp.empty(num_tets, dtype=float, device=device)
# 四面体应变能
tet_energies=wp.empty(num_tets, dtype=float, device=device, requires_grad=True)

# 参考边矩阵
Dm=wp.empty(num_tets, dtype=wp.mat33, device=device)
# Dm的逆
Dm_inv=wp.empty(num_tets, dtype=wp.mat33, device=device)
# 变形边矩阵
Ds=wp.empty(num_tets, dtype=wp.mat33, device=device)
# 变形梯度
F=wp.empty(num_tets, dtype=wp.mat33, device=device)

# 四面体全部面
all_faces=wp.empty(num_tets*4, dtype=wp.vec3i, device=device)

# Barrier constant
d_tilde=1e-3
kappa = 1e5

wp.launch(
    kernel=geometry.extract_all_faces,
    dim=num_tets,
    inputs=[tet_indices],
    outputs=[all_faces],
    device=device,
)

surface_faces, surface_edges=geometry.extract_surface_faces_and_edges(all_faces)
surface_faces_np=surface_faces.numpy()
surface_edges_np=surface_edges.numpy()
PT_pair=geometry.make_PT_candidates(surface_faces_np, device=device)
EE_pair=geometry.make_EE_candidates(surface_edges_np, device=device)

PT_barrier_energies=wp.empty(
    PT_pair.shape[0],
    dtype=float,
    device=device,
    requires_grad=True,
)
EE_barrier_energies=wp.empty(
    EE_pair.shape[0],
    dtype=float,
    device=device,
    requires_grad=True,
)

# 三类能量分别归约为标量，最后再合并为总能量
elastic_total=wp.zeros(1, dtype=float, device=device, requires_grad=True)
PT_total=wp.zeros(1, dtype=float, device=device, requires_grad=True)
EE_total=wp.zeros(1, dtype=float, device=device, requires_grad=True)
total_energy=wp.zeros(1, dtype=float, device=device, requires_grad=True)
tape=wp.Tape()

wp.launch(
    kernel=energy.compute_rest_data,
    dim=num_tets,
    inputs=[
        tet_indices,
        rest_positions,
    ],
    outputs=[
        Dm,
        Dm_inv,
        rest_volumes,
    ],
    device=device,
)

with tape:


    '''计算弹性能'''
    wp.launch(
        kernel=energy.compute_spatial_data,
        dim=num_tets,
        inputs=[
            tet_indices,
            positions,
            Dm_inv,
            rest_volumes,
            mu,
            lamda,
        ],
        outputs=[
            Ds,
            F,
            tet_energies,
        ],
        device=device,
    )

    '''计算 Barrier'''
    energy.compute_barrier_energies(
        positions,
        PT_pair,
        EE_pair,
        d_tilde,
        kappa,
        PT_barrier_energies,
        EE_barrier_energies
    )

    '''合并能量'''
    wp.launch(
        kernel=energy.reduce_energy,
        dim=num_tets,
        inputs=[
            tet_energies,
        ],
        outputs=[
            elastic_total,
        ],
        device=device,
    )

    if PT_pair.shape[0] > 0:
        wp.launch(
            kernel=energy.reduce_energy,
            dim=PT_pair.shape[0],
            inputs=[
                PT_barrier_energies,
            ],
            outputs=[
                PT_total,
            ],
            device=device,
        )

    if EE_pair.shape[0] > 0:
        wp.launch(
            kernel=energy.reduce_energy,
            dim=EE_pair.shape[0],
            inputs=[
                EE_barrier_energies,
            ],
            outputs=[
                EE_total,
            ],
            device=device,
        )

    wp.launch(
        kernel=energy.combine_energy,
        dim=1,  
        inputs=[
            elastic_total,
            PT_total,
            EE_total,
        ],
        outputs=[
            total_energy,
        ],
        device=device,
    )

tape.backward(loss=total_energy)

gradient_np = positions.grad.numpy()