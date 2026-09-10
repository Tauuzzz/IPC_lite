from typing import Tuple

import warp as wp
import numpy as np


# ---------------------------------------------------------------------------
# 规则四面体网格生成（学习用 demo 网格）
# ---------------------------------------------------------------------------

def build_regular_tet_grid(nc: int, size: float = 0.2):
    """生成 nc x nc x nc 个 cell 的规则四面体网格（每个 cube 拆 6 个 tet）。

    Returns:
        verts_np: ((nc+1)^3, 3) float32 顶点坐标，按 (i, j, k) 顺序
        tets_np:  (6*nc^3, 4) int32 tet 顶点索引
        pinned_np: ((nc+1)^3,) bool，z=0 层固定
    """
    nv = nc + 1
    h = size / nc

    # 顶点 (i, j, k) -> index i + j*nv + k*nv*nv
    verts = np.zeros((nv * nv * nv, 3), dtype=np.float32)
    for k in range(nv):
        for j in range(nv):
            for i in range(nv):
                verts[i + j * nv + k * nv * nv] = (i * h, j * h, k * h)

    # 每个 cell 拆成 6 个 tet：绕主对角线 (a, a2, b2, d2) 的 6 个非退化 tet。
    # 局部角点编号：0=a 1=b 2=c 3=d 4=a2 5=b2 6=c2 7=d2
    #   a=(0,0,0) b=(1,0,0) c=(0,1,0) d=(1,1,0)
    #   a2=(0,0,1) b2=(1,0,1) c2=(0,1,1) d2=(1,1,1)
    tets = []
    for k in range(nc):
        for j in range(nc):
            for i in range(nc):
                base = i + j * nv + k * nv * nv
                v = [
                    base,            # 0 a
                    base + 1,        # 1 b
                    base + nv,       # 2 c
                    base + nv + 1,   # 3 d
                    base + nv * nv,          # 4 a2
                    base + 1 + nv * nv,      # 5 b2
                    base + nv + nv * nv,     # 6 c2
                    base + nv + 1 + nv * nv, # 7 d2
                ]
                for tet in [(0,1,3,7), (0,1,7,5), (0,2,7,3), (0,2,6,7), (0,4,5,7), (0,4,7,6)]:
                    tets.append(tuple(v[idx] for idx in tet))

    tets_np = np.asarray(tets, dtype=np.int32)

    # 固定 z=0 层
    pinned_np = np.zeros(nv * nv * nv, dtype=bool)
    for j in range(nv):
        for i in range(nv):
            pinned_np[i + j * nv] = True

    return verts, tets_np, pinned_np


def build_two_cubes(nc: int = 2, size: float = 0.2, gap: float = 0.03, lateral: float = 0.02):
    """上下两块相隔 gap 的立方体：底部固定，顶部带横向偏移。

    跨物体的 PT/EE 对是正常对（初始距离 = gap > 0），不会被候选过滤删掉，
    互相靠近到 d < d_tilde 时 barrier 激活；顶部横向滑移激发摩擦。

    Returns:
        verts_np, tets_np, pinned_np（同 build_regular_tet_grid）
    """
    verts0, tets0, pinned0 = build_regular_tet_grid(nc, size)
    verts1, tets1, _ = build_regular_tet_grid(nc, size)

    # 上块：抬高 size+gap，x 方向横移 lateral
    verts1 = verts1.astype(np.float64)
    verts1[:, 0] += lateral
    verts1[:, 2] += size + gap

    offset = verts0.shape[0]
    verts = np.concatenate([verts0, verts1.astype(np.float32)])
    tets = np.concatenate([tets0, tets1 + offset])
    pinned = np.concatenate([pinned0, np.zeros(verts1.shape[0], dtype=bool)])
    return verts, tets, pinned


def filter_degenerate_candidates(
    surface_faces_np,
    surface_edges_np,
    rest_positions_np,
    threshold_sq: float = 1e-12,
    device: str = "cpu",
):
    """去掉初始构型上距离严格为 0 的 PT/EE 候选对。

    固定候选集合的设计里，同一表面相邻（共面、共享边）的三角形之间
    会出现"顶点 vs 相邻面"距离 = 0 的退化对（例如正方形沿对角线剖成
    两个三角形，对角的顶点投影落在共享对角线上）。barrier 在 d=0 处
    发散，必须把这些对提前剔除。真实 IPC 每 step 重建候选集合，
    用 broad-phase + 距离阈值就天然不会产生这种对。

    注意：这里只剔距离 == 0 的退化为，阈值和 barrier 的 d_tilde 无关，
    否则会把初始距离在 d_tilde 以内的正常对（比如两个快要接触的物体）
    也删掉，接触就永远触发不了了。
    """
    verts = wp.array(rest_positions_np.astype(np.float32), dtype=wp.vec3, device=device)
    PT_pair = make_PT_candidates(surface_faces_np, device=device)
    EE_pair = make_EE_candidates(surface_edges_np, device=device)

    PT_distances_squared, EE_distances_squared, _, _ = compute_distance(
        verts, PT_pair, EE_pair, max(1e-9, threshold_sq**0.5)
    )
    PT_keep = PT_distances_squared.numpy() > threshold_sq
    EE_keep = EE_distances_squared.numpy() > threshold_sq

    PT_pair = wp.array(PT_pair.numpy()[PT_keep], dtype=wp.vec4i, device=device)
    EE_pair = wp.array(EE_pair.numpy()[EE_keep], dtype=wp.vec4i, device=device)
    return PT_pair, EE_pair

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


@wp.func
def point_triangle_closest_point(
    point: wp.vec3,
    A: wp.vec3,
    B: wp.vec3,
    C: wp.vec3,
) -> wp.vec2:
    # 点到三角形平面的最近点重心坐标 (beta1, beta2)
    # 最近点 = A + beta1 * (B - A) + beta2 * (C - A)
    # 这里不做 clamp：投影落在三角形外时也直接返回，和 IPC toolkit 一致
    e0 = B - A
    e1 = C - A
    AP = point - A

    a00 = wp.dot(e0, e0)
    a01 = wp.dot(e0, e1)
    a11 = wp.dot(e1, e1)

    b0 = wp.dot(e0, AP)
    b1 = wp.dot(e1, AP)

    det = a00 * a11 - a01 * a01

    # 三角形退化
    if wp.abs(det) <= 1.0e-12:
        return wp.vec2(0.0, 0.0)

    beta1 = (a11 * b0 - a01 * b1) / det
    beta2 = (a00 * b1 - a01 * b0) / det
    return wp.vec2(beta1, beta2)


@wp.func
def edge_edge_closest_point(
    A: wp.vec3,
    B: wp.vec3,
    C: wp.vec3,
    D: wp.vec3,
) -> wp.vec2:
    # 两条边最近点的参数 (alpha1, alpha2)
    # 最近点分别是 A + alpha1 * (B - A) 和 C + alpha2 * (D - C)
    # 和 compute_edge_to_edge_distance_squared 里的 s、t 是同一组量
    ea = B - A
    eb = D - C
    w = A - C

    a = wp.dot(ea, ea)
    b = wp.dot(ea, eb)
    c = wp.dot(eb, eb)
    d = wp.dot(ea, w)
    e = wp.dot(eb, w)

    det = a * c - b * b

    # 两条边平行或退化
    if wp.abs(det) <= 1.0e-12:
        return wp.vec2(0.0, 0.0)

    alpha1 = (b * e - c * d) / det
    alpha2 = (a * e - b * d) / det
    return wp.vec2(alpha1, alpha2)


@wp.func
def point_triangle_tangent_basis(
    A: wp.vec3,
    B: wp.vec3,
    C: wp.vec3,
) -> Tuple[wp.vec3, wp.vec3]:
    # 三角形所在平面的正交单位切向基 (t0, t1)
    # t0 沿第一条边，t1 在平面内且垂直于 t0
    e0 = B - A
    e0_length_squared = wp.dot(e0, e0)

    # 三角形退化，定义不出切平面
    if e0_length_squared <= 1.0e-24:
        return wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)

    t0 = e0 / wp.sqrt(e0_length_squared)

    normal = wp.cross(e0, C - A)
    t1_raw = wp.cross(normal, e0)
    t1_length_squared = wp.dot(t1_raw, t1_raw)

    # C 落在 AB 上，三角形退化为线段，只剩一个切方向
    if t1_length_squared <= 1.0e-24:
        return t0, wp.vec3(0.0, 0.0, 0.0)

    return t0, t1_raw / wp.sqrt(t1_length_squared)


@wp.func
def edge_edge_tangent_basis(
    A: wp.vec3,
    B: wp.vec3,
    C: wp.vec3,
    D: wp.vec3,
) -> Tuple[wp.vec3, wp.vec3]:
    # 两条边张成的平面的正交单位切向基 (t0, t1)
    ea = B - A
    eb = D - C
    ea_length_squared = wp.dot(ea, ea)

    if ea_length_squared <= 1.0e-24:
        return wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)

    t0 = ea / wp.sqrt(ea_length_squared)

    # 接触法向 n = ea x eb
    normal = wp.cross(ea, eb)
    t1_raw = wp.cross(normal, ea)
    t1_length_squared = wp.dot(t1_raw, t1_raw)

    # 两边平行时 normal = 0，滑动只可能沿边方向，t1 置零即可
    if t1_length_squared <= 1.0e-24:
        return t0, wp.vec3(0.0, 0.0, 0.0)

    return t0, t1_raw / wp.sqrt(t1_length_squared)


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


