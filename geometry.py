import warp as wp
import numpy as np

@wp.kernel
def extract_all_faces(tet_indices: wp.array[wp.vec4i],
                          surface_faces: wp.array[wp.vec3i]
                          ):
    tet_index = wp.tid()
    tet = tet_indices[tet_index]
    # 提取顶点索引
    i0 = tet[0]
    i1 = tet[1]
    i2 = tet[2]
    i3 = tet[3]
    # 四面体的四个面(这里还没有去重)
    surface_faces[tet_index * 4 + 0] = wp.vec3i(i0, i1, i2)
    surface_faces[tet_index * 4 + 1] = wp.vec3i(i0, i1, i3)
    surface_faces[tet_index * 4 + 2] = wp.vec3i(i0, i2, i3)
    surface_faces[tet_index * 4 + 3] = wp.vec3i(i1, i2, i3)

def extract_surface_faces_and_edges(surface_faces: wp.array[wp.vec3i]):
    all_faces_np=surface_faces.numpy()
    # 将所有面排序，以便去重
    sorted_faces_np=np.sort(all_faces_np,axis=1)

    unique_faces_np, counts = np.unique(
        sorted_faces_np,
        axis=0,
        return_counts=True,
    )

    # 只保留出现一次的外表面
    surface_faces_np = unique_faces_np[counts == 1]
    # 提取所有边
    edges = []
    for face in surface_faces_np:
        edges.append((face[0], face[1]))
        edges.append((face[0], face[2]))
        edges.append((face[1], face[2]))

    surface_edges_np = np.array(edges, dtype=np.int32)
    # 将边排序，以便去重
    sorted_edges_np = np.sort(surface_edges_np, axis=1)

    unique_edges_np = np.unique(
        sorted_edges_np,
        axis=0
    )

    surface_edges = wp.array(
        unique_edges_np,
        dtype=wp.vec2i,
        device=surface_faces.device,
    )

    surface_faces = wp.array(
        surface_faces_np,
        dtype=wp.vec3i,
        device=surface_faces.device,
    )

    return surface_faces, surface_edges

def make_PT_candidates(surface_faces_np, device=None):
    surface_vertices_np = np.unique(surface_faces_np.reshape(-1))
    PT_candidates = []

    for vertex in surface_vertices_np:
        # 不包含该顶点的面
        other_faces = surface_faces_np[np.all(surface_faces_np != vertex, axis=1)]
        for face in other_faces:
            PT_candidates.append((vertex, face[0], face[1], face[2]))

    # reshape保证没有候选对时，数组形状仍然是(0, 4)
    PT_candidates_np = np.asarray(
        PT_candidates,
        dtype=np.int32,
    ).reshape(-1, 4)

    return wp.array(
        PT_candidates_np,
        dtype=wp.vec4i,
        device=device,
    )


def make_EE_candidates(surface_edges_np, device=None):
    EE_candidates = []
    num_edges = surface_edges_np.shape[0]

    for i in range(num_edges):
        edge0 = surface_edges_np[i]
        for j in range(i + 1, num_edges):
            edge1 = surface_edges_np[j]
            # 如果两个边没有公共顶点，则它们是EE候选对
            has_shared_vertex = (
                (edge0[0] == edge1[0])
                or (edge0[0] == edge1[1])
                or (edge0[1] == edge1[0])
                or (edge0[1] == edge1[1])
            )

            if not has_shared_vertex:
                EE_candidates.append((edge0[0], edge0[1], edge1[0], edge1[1]))

    # reshape保证没有候选对时，数组形状仍然是(0, 4)
    EE_candidates_np = np.asarray(
        EE_candidates,
        dtype=np.int32,
    ).reshape(-1, 4)

    return wp.array(
        EE_candidates_np,
        dtype=wp.vec4i,
        device=device,
    )


@wp.func
def is_point_in_triangle(
    point: wp.vec3,
    A: wp.vec3,
    B: wp.vec3,
    C: wp.vec3,
) -> bool:
    # 计算向量
    AB = B - A
    AC = C - A
    AP = point - A

    # 计算点积
    dotABAB = wp.dot(AB, AB)
    dotABAC = wp.dot(AB, AC)
    dotACAC = wp.dot(AC, AC)
    dotAPAB = wp.dot(AP, AB)
    dotAPAC = wp.dot(AP, AC)

    # 计算重心坐标
    denom = dotABAB * dotACAC - dotABAC * dotABAC
    if denom <= 1.0e-12:
        return False  # 三角形退化为一条线段

    u = (dotACAC * dotAPAB - dotABAC * dotAPAC) / denom
    v = (dotABAB * dotAPAC - dotABAC * dotAPAB) / denom

    eps = 1.0e-6
    return (u >= -eps) and (v >= -eps) and (u + v <= 1.0 + eps)


@wp.func
def compute_point_to_segment_distance_squared(
    point: wp.vec3,
    A: wp.vec3,
    B: wp.vec3,
) -> float:
    AB = B - A
    AP = point - A

    length_squared = wp.dot(AB, AB)

    # 线段退化为一个点
    if length_squared <= 1.0e-12:
        return wp.dot(AP, AP)

    # 计算点到线段的投影
    t = wp.dot(AP, AB) / length_squared
    t = wp.clamp(t, 0.0, 1.0)

    projection = A + t * AB
    difference = point - projection
    return wp.dot(difference, difference)


@wp.func
def compute_point_to_triangle_distance_squared(
    point: wp.vec3,
    A: wp.vec3,
    B: wp.vec3,
    C: wp.vec3,
) -> float:
    # 计算向量
    AB = B - A
    AC = C - A
    AP = point - A

    # 计算三角形的法向量
    N = wp.cross(AB, AC)
    N_length_squared = wp.dot(N, N)

    if N_length_squared <= 1.0e-12:
        # 三角形退化时，取点到三条边的最小平方距离
        d1 = compute_point_to_segment_distance_squared(point, A, B)
        d2 = compute_point_to_segment_distance_squared(point, B, C)
        d3 = compute_point_to_segment_distance_squared(point, C, A)
        return wp.min(d1, wp.min(d2, d3))

    # 计算点到三角形平面的投影
    normal_projection = wp.dot(AP, N)
    projection = point - normal_projection / N_length_squared * N

    # 检查投影是否在三角形内
    if is_point_in_triangle(projection, A, B, C):
        return normal_projection * normal_projection / N_length_squared
    else:
        # 如果不在三角形内，返回点到三条边的最小平方距离
        d1 = compute_point_to_segment_distance_squared(point, A, B)
        d2 = compute_point_to_segment_distance_squared(point, B, C)
        d3 = compute_point_to_segment_distance_squared(point, C, A)
        return wp.min(d1, wp.min(d2, d3))


@wp.func
def compute_edge_to_edge_distance_squared(
    A: wp.vec3,
    B: wp.vec3,
    C: wp.vec3,
    D: wp.vec3,
) -> float:
    u = B - A
    v = D - C
    w = A - C

    a = wp.dot(u, u)
    b = wp.dot(u, v)
    c = wp.dot(v, v)
    d = wp.dot(u, w)
    e = wp.dot(v, w)

    # 两条边都退化为点
    if a <= 1.0e-12 and c <= 1.0e-12:
        return wp.dot(A - C, A - C)

    # 第一条边退化为点
    if a <= 1.0e-12:
        return compute_point_to_segment_distance_squared(A, C, D)

    # 第二条边退化为点
    if c <= 1.0e-12:
        return compute_point_to_segment_distance_squared(C, A, B)

    delta = a * c - b * b

    # 非平行时，先检查无限直线的最近点是否同时位于两条边内部
    if delta > 1.0e-12 * a * c:
        s = (b * e - c * d) / delta
        t = (a * e - b * d) / delta

        if s >= 0.0 and s <= 1.0 and t >= 0.0 and t <= 1.0:
            closest_point_edge0 = A + s * u
            closest_point_edge1 = C + t * v
            difference = closest_point_edge0 - closest_point_edge1
            return wp.dot(difference, difference)

    # 平行或无限直线最近点越界时，最近点位于参数区域的边界
    d1 = compute_point_to_segment_distance_squared(A, C, D)
    d2 = compute_point_to_segment_distance_squared(B, C, D)
    d3 = compute_point_to_segment_distance_squared(C, A, B)
    d4 = compute_point_to_segment_distance_squared(D, A, B)
    return wp.min(wp.min(d1, d2), wp.min(d3, d4))


@wp.kernel
def compute_PT_distance_kernel(
    positions: wp.array[wp.vec3],
    PT_pair: wp.array[wp.vec4i],
    d_tilde_squared: float,
    PT_distances_squared: wp.array[float],
    PT_close_pairs: wp.array[bool],
):
    pair_index = wp.tid()
    pair = PT_pair[pair_index]

    point = positions[pair[0]]
    A = positions[pair[1]]
    B = positions[pair[2]]
    C = positions[pair[3]]

    distance_squared = compute_point_to_triangle_distance_squared(
        point,
        A,
        B,
        C,
    )
    PT_distances_squared[pair_index] = distance_squared
    PT_close_pairs[pair_index] = distance_squared < d_tilde_squared


@wp.kernel
def compute_EE_distance_kernel(
    positions: wp.array[wp.vec3],
    EE_pair: wp.array[wp.vec4i],
    d_tilde_squared: float,
    EE_distances_squared: wp.array[float],
    EE_close_pairs: wp.array[bool],
):
    pair_index = wp.tid()
    pair = EE_pair[pair_index]

    A = positions[pair[0]]
    B = positions[pair[1]]
    C = positions[pair[2]]
    D = positions[pair[3]]

    distance_squared = compute_edge_to_edge_distance_squared(A, B, C, D)
    EE_distances_squared[pair_index] = distance_squared
    EE_close_pairs[pair_index] = distance_squared < d_tilde_squared


def compute_distance(positions, PT_pair, EE_pair, d_tilde):
    """计算全部PT/EE候选对的平方距离和接近标记。"""

    if d_tilde <= 0.0:
        raise ValueError("d_tilde必须大于0")

    num_PT_pairs = PT_pair.shape[0]
    num_EE_pairs = EE_pair.shape[0]
    device = positions.device

    if PT_pair.device != device or EE_pair.device != device:
        raise ValueError("positions、PT_pair和EE_pair必须位于同一个Warp设备")

    d_tilde_squared = d_tilde * d_tilde

    PT_distances_squared = wp.empty(
        num_PT_pairs,
        dtype=float,
        device=device,
    )
    EE_distances_squared = wp.empty(
        num_EE_pairs,
        dtype=float,
        device=device,
    )
    PT_close_pairs = wp.empty(
        num_PT_pairs,
        dtype=bool,
        device=device,
    )
    EE_close_pairs = wp.empty(
        num_EE_pairs,
        dtype=bool,
        device=device,
    )

    if num_PT_pairs > 0:
        wp.launch(
            kernel=compute_PT_distance_kernel,
            dim=num_PT_pairs,
            inputs=[
                positions,
                PT_pair,
                d_tilde_squared,
                PT_distances_squared,
                PT_close_pairs,
            ],
            device=device,
        )

    if num_EE_pairs > 0:
        wp.launch(
            kernel=compute_EE_distance_kernel,
            dim=num_EE_pairs,
            inputs=[
                positions,
                EE_pair,
                d_tilde_squared,
                EE_distances_squared,
                EE_close_pairs,
            ],
            device=device,
        )

    return (
        PT_distances_squared,
        EE_distances_squared,
        PT_close_pairs,
        EE_close_pairs,
    )


