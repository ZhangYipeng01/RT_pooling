import re
import numpy as np
from itertools import combinations
import plotly.graph_objects as go
from mayavi import mlab
from scipy.spatial import ConvexHull
import ast

def TXTtoList(filepath, return_type='str'):
    fp = open(filepath, 'r', encoding='utf-8')
    content = fp.read()
    fp.close()
    rowlist = content.splitlines()
    recordlist = [row.split() for row in rowlist if row != 'END' if row.strip()]
    if return_type == 'str':
        return recordlist
    if return_type == 'int':
        int_recordlist = [[int(element) for element in inner_list] for inner_list in recordlist]
        return int_recordlist
    if return_type == 'float':
        float_recordlist = [[float(element) for element in inner_list] for inner_list in recordlist]
        return float_recordlist

def get_delaunay(file_path):
    # 初始化存储变量
    dslice = []
    current_slice = None

    # 打开并读取文件
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            # 检测当前行属于哪个slice
            if 'Slice' in line:
                # 移除冒号并获取slice编号
                slice_number = int(line.split(' ')[1].replace(':', ''))
                current_slice = slice_number
                dslice.append([])
            else:
                # 解析数据和值
                data, value = eval(line)
                dslice[current_slice-1].append(data)


    # 返回处理后的数据
    return dslice

def vertex_get_coords(vertex):
    if len(points_set[0]) == 2:
        coords = np.array([0.0, 0.0, 0.0])
    if len(points_set[0]) == 3:
        coords = np.array([0.0, 0.0, 0.0, 0.0])
    for index in vertex:
        coords = np.array(coords) + np.array(points_set[index]+[-1.0])
    return coords.tolist()


# 绘制 3D 凸包
def plot_3d_convex_hull(points, labels=None, edges=None):
    # 检查输入点
    points = np.array(points)  # 确保是 NumPy 数组
    if len(points) < 4:  # 确保至少有 4 个点才能构成凸包
        raise ValueError("至少需要 4 个点来构造 3D 凸包")

    # 计算凸包
    hull = ConvexHull(points)

    # 创建 Mayavi 图形窗口
    mlab.figure(size=(800, 600), bgcolor=(1, 1, 1))  # 设置背景为白色

    # 绘制凸包表面
    for simplex in hull.simplices:
        triangle_points = points[simplex]
        x, y, z = triangle_points.T
        mlab.triangular_mesh(x, y, z, [[0, 1, 2]], color=(0.5, 0.8, 1), opacity=0.5)

    # 绘制所有边
    if edges is not None:
        for edge in edges:
            x, y, z = np.array(edge).T
            mlab.plot3d(x, y, z, color=(1, 0, 0), tube_radius=0.01)

    # 添加顶点和标签
    mlab.points3d(points[:, 0], points[:, 1], points[:, 2],
                  color=(0, 0, 0), scale_factor=0.02)  # 顶点为黑色小球
    if labels:
        for i, label in enumerate(labels):
            x, y, z = points[i]
            # 调整文字比例，增加标签清晰度
            mlab.text3d(
                x, y, z, str(label),
                scale=(0.1, 0.1, 0.1),  # 调整文字大小
                color=(0, 0, 0)  # 设置文字颜色为黑色
            )

    mlab.show()


def plot_Delaunay(slice):
    fig = go.Figure()
    v = []
    v_labels = []
    e = []
    t = []
    for simplex in slice:
        if len(simplex) == 1:
            v_labels.append(str(simplex[0]))
            v.append([vertex_get_coords(simplex[0])[:2]])
        if len(simplex) == 2:
            e.append([vertex_get_coords(simplex[0])[:2], vertex_get_coords(simplex[1])[:2]])
        if len(simplex) == 3:
            t.append([vertex_get_coords(simplex[0])[:2], vertex_get_coords(simplex[1])[:2],
                      vertex_get_coords(simplex[2])[:2]])
    print(v)
    print(v_labels)

    # 绘制三角形
    for triangle in t:
        x, y = zip(*triangle)  # 解包顶点坐标
        x += (x[0],)  # 闭合三角形
        y += (y[0],)
        fig.add_trace(go.Scatter(
            x=x, y=y, fill='toself', mode='lines',
            line=dict(color='black'), fillcolor='lightblue', opacity=0.6
        ))

    # 绘制边
    for edge in e:
        x, y = zip(*edge)  # 解包顶点坐标
        fig.add_trace(go.Scatter(
            x=x, y=y, mode='lines', line=dict(color='red', width=2)  # 边的粗细为 2
        ))

    # 绘制点
    for i in range(len(v)):
        x, y = zip(*v[i])  # 获取所有点的 x 和 y 坐标
        fig.add_trace(go.Scatter(
            x=x, y=y, mode='markers+text',  # 同时显示点和标签
            marker=dict(size=12, color='red'),  # 点的大小为 8（边的粗细为 2 的两倍）
            #text=v_labels[i],  # 设置点的标签
            textposition="top center",  # 标签位置
            textfont=dict(size=12, color="black")  # 标签字体样式
        ))

    # 更新布局：隐藏坐标轴，锁定比例一致，背景纯白
    fig.update_layout(
        xaxis=dict(
            visible=False,  # 隐藏 X 轴
            scaleanchor="y"  # 锁定 X 和 Y 轴比例一致
        ),
        yaxis=dict(
            visible=False  # 隐藏 Y 轴
        ),
        plot_bgcolor="white",  # 设置背景为纯白
        paper_bgcolor="white",  # 设置图表外部为纯白
        showlegend=False  # 隐藏图例
    )

    fig.show()

def plot_rhomboidtiling(rhomboids, all_points, all_labels=None, edges=None, edges_Del = None, triangle_Del = None):
    # 创建 Mayavi 图形窗口
    mlab.figure(size=(800, 600), bgcolor=(1, 1, 1))  # 设置背景为白色

    # 绘制每个 rhomboid 的凸包
    for points in rhomboids:
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            triangle_points = np.array(points)[simplex]
            x, y, z = triangle_points.T
            mlab.triangular_mesh(x, y, z, [[0, 1, 2]], color=(0.8, 0.95, 1), opacity=0.1)

    # 绘制所有边
    if edges is not None:
        for edge in edges:
            x, y, z = np.array(edge).T
            mlab.plot3d(x, y, z, color=(1, 0, 0), tube_radius=0.01, opacity=0.3)

    for edge in edges_Del:
        x, y, z = np.array(edge).T
        mlab.plot3d(x, y, z, color=(1, 0, 0), tube_radius=0.01)

    # 绘制三角形
    for triangle in triangle_Del:
        x, y, z = np.array(triangle).T
        mlab.triangular_mesh(
            x, y, z,
            [[0, 1, 2]],
            color=(0.7, 0.9, 1),
            opacity=0.9
        )

    # 绘制固定的 z=-1, z=-2, z=-3 平面
    z_levels = [-1, -2, -3]
    x_range = [all_points[:, 0].min()-1, all_points[:, 0].max()+1]
    y_range = [all_points[:, 1].min()-1, all_points[:, 1].max()+1]

    for z in z_levels:
        x = np.linspace(x_range[0], x_range[1], 50)
        y = np.linspace(y_range[0], y_range[1], 50)
        x, y = np.meshgrid(x, y)
        z_plane = np.full_like(x, z)
        mlab.mesh(x, y, z_plane, color=(0.8, 0.8, 0.8), opacity=0.5)  # 灰色透明平面

    # 绘制顶点并禁用阴影
    glyphs = mlab.points3d(
        all_points[:, 0], all_points[:, 1], all_points[:, 2],
        mode='sphere',  # 设置点的显示模式为球体
        color=(0.8, 0, 0),  # 统一设置颜色为红色
        scale_factor=0.1,  # 统一设置大小
        scale_mode='none'  # 禁用标量值对大小的影响
    )
    # 禁用顶点的光照
    glyphs.actor.property.lighting = False

    # 设置相机为正交投影
    scene = mlab.gcf().scene  # 获取当前场景
    scene.camera.parallel_projection = True  # 开启正交投影

    camera = scene.camera
    camera.position = [7.62315739, 10.20062359, 1.32246539]
    camera.focal_point = [0, 0.25, -2.00000009]
    camera.view_angle = 30.0
    camera.view_up = [-0.19601296, -0.17216689, 0.96536909]

    # 可选：添加标签
    # if all_labels:
    #     for i, label in enumerate(all_labels):
    #         x, y, z = all_points[i]
    #         mlab.text3d(
    #             x, y, z, str(label),
    #             scale=(0.15, 0.15, 0.15),  # 增大标签大小
    #             color=(0, 0, 0)  # 黑色字体
    #         )

    mlab.show()

def plot_one_rhomboid_with_rhomboidtiling(rhomboid, rhomboids, all_points, all_labels=None, edges=None, edges_Del = None, triangle_Del = None):
    # 创建 Mayavi 图形窗口
    mlab.figure(size=(800, 600), bgcolor=(1, 1, 1))  # 设置背景为白色


    # 绘制每个 rhomboid 的凸包
    for points in rhomboids:
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            triangle_points = np.array(points)[simplex]
            x, y, z = triangle_points.T
            mlab.triangular_mesh(x, y, z, [[0, 1, 2]], color=(0.8, 0.95, 1), opacity=0.1)

    hull2 = ConvexHull(rhomboid)
    for simplex in hull2.simplices:
        triangle_points = np.array(rhomboid)[simplex]
        x, y, z = triangle_points.T
        mlab.triangular_mesh(x, y, z, [[0, 1, 2]], color=(0, 1, 0.5), opacity=0.2)

    # 绘制所有边
    if edges is not None:
        for edge in edges:
            x, y, z = np.array(edge).T
            mlab.plot3d(x, y, z, color=(1, 0, 0), tube_radius=0.01, opacity=0.3)

    for edge in edges_Del:
        x, y, z = np.array(edge).T
        mlab.plot3d(x, y, z, color=(1, 0, 0), tube_radius=0.01)

    # 绘制三角形
    for triangle in triangle_Del:
        x, y, z = np.array(triangle).T
        mlab.triangular_mesh(
            x, y, z,
            [[0, 1, 2]],
            color=(0.7, 0.9, 1),
            opacity=0.9
        )

    # 绘制固定的 z=-1, z=-2, z=-3 平面
    z_levels = [-1, -2, -3]
    x_range = [all_points[:, 0].min()-1, all_points[:, 0].max()+1]
    y_range = [all_points[:, 1].min()-1, all_points[:, 1].max()+1]

    for z in z_levels:
        x = np.linspace(x_range[0], x_range[1], 50)
        y = np.linspace(y_range[0], y_range[1], 50)
        x, y = np.meshgrid(x, y)
        z_plane = np.full_like(x, z)
        mlab.mesh(x, y, z_plane, color=(0.8, 0.8, 0.8), opacity=0.5)  # 灰色透明平面

    # 绘制顶点并禁用阴影
    glyphs = mlab.points3d(
        all_points[:, 0], all_points[:, 1], all_points[:, 2],
        mode='sphere',  # 设置点的显示模式为球体
        color=(0.8, 0, 0),  # 统一设置颜色为红色
        scale_factor=0.1,  # 统一设置大小
        scale_mode='none'  # 禁用标量值对大小的影响
    )
    # 禁用顶点的光照
    glyphs.actor.property.lighting = False

    # 设置相机为正交投影
    scene = mlab.gcf().scene  # 获取当前场景
    scene.camera.parallel_projection = True  # 开启正交投影

    camera = scene.camera
    camera.position = [7.62315739, 10.20062359, 1.32246539]
    camera.focal_point = [0, 0.25, -2.00000009]
    camera.view_angle = 30.0
    camera.view_up = [-0.19601296, -0.17216689, 0.96536909]

    # 可选：添加标签
    # if all_labels:
    #     for i, label in enumerate(all_labels):
    #         x, y, z = all_points[i]
    #         mlab.text3d(
    #             x, y, z, str(label),
    #             scale=(0.15, 0.15, 0.15),  # 增大标签大小
    #             color=(0, 0, 0)  # 黑色字体
    #         )

    mlab.show()


points_set = TXTtoList('4points.txt','float')
print(points_set)

file_path = '4points_fslices.txt'
Delaunay = get_delaunay(file_path)
edges_Del = []
triangles_Del = []
for slice in Delaunay:
    plot_Delaunay(slice)
    for simplex in slice:
        if len(simplex) == 2:
            edges_Del.append([vertex_get_coords(simplex[0]),vertex_get_coords(simplex[1])])
        if len(simplex) == 3:
            triangles_Del.append([vertex_get_coords(simplex[0]),vertex_get_coords(simplex[1]),vertex_get_coords(simplex[2])])



# 初始化一个空列表来存储edges_core
edges_core = []
with open('4points_rhomboidtiling.txt', 'r') as file:
    for line in file:
        if 'd:1' in line:
            # 使用正则表达式匹配 vxs: 后的列表数据
            match = re.search(r'vxs:\s*(\[\[.*?\]\])', line)
            if match:
                # 获取匹配到的列表字符串
                vxs_data = match.group(1)
                # 由于我们不再使用 eval(), 这里可以采用一个简单的方法来解析这个字符串
                # 注意：这种方法假设数据格式非常规范，仅适用于格式良好的数据
                # 对于更复杂或不规范的数据格式，可能需要更复杂的解析方法
                try:
                    edges = ast.literal_eval(vxs_data)
                    edges_core.append(edges)
                except ValueError as e:
                    print(f"Error parsing vxs_data: {vxs_data}")
                    print(e)

edges_coord_set = []
for point in points_set:
    edges_coord_set.append([[0,0,0],point+[-1]])
for edge_indexes in edges_core:
    v_0 = vertex_get_coords(edge_indexes[0])
    v_1 = vertex_get_coords(edge_indexes[1])
    edges_coord_set.append([v_0,v_1])

rhomboid_core = []
# 打开并读取文件
with open('4points_rhomboids.txt', 'r') as file:
    for line in file:
        line = line.strip()
        lists = re.findall(r'\[(.*?)\]', line)

        temp_result = []
        for list_str in lists:
            if list_str:
                list_int = [int(x) for x in list_str.split(',')]
            else:
                list_int = []
            temp_result.append(list_int)
        rhomboid_core.append(temp_result)

rhomboids_tiling_vertexes_set = []
all_vertexes = []
labels_set = []
all_labels_set = []
for rhomboid in rhomboid_core:
    labels = []
    rhomboid_vertexes_set = []
    X_in = rhomboid[0]
    X_on = rhomboid[1]
    v_X_in = vertex_get_coords(X_in)
    if len(X_in) == 0:
        labels.append('O')
    else:
        labels.append(str(X_in))
    rhomboid_vertexes_set.append(v_X_in)
    if v_X_in not in all_vertexes:
        if len(X_in) == 0:
            all_labels_set.append('O')
        else:
            all_labels_set.append(str(X_in))
        all_vertexes.append(v_X_in)

    for r in range(len(X_on)):
        for subset in combinations(X_on, r+1):
            if len(X_in) == 0:
                labels.append(str(list(subset)))
            else:
                labels.append(str(X_in+list(subset)))
            root_point = np.array(v_X_in) + np.array(vertex_get_coords(subset))
            rhomboid_vertexes_set.append(root_point.tolist())
            if root_point.tolist() not in all_vertexes:
                if len(X_in) == 0:
                    all_labels_set.append(str(list(subset)))
                else:
                    all_labels_set.append(str(X_in+list(subset)))
                all_vertexes.append(root_point.tolist())

    rhomboids_tiling_vertexes_set.append(rhomboid_vertexes_set)
    labels_set.append(labels)

# for i in range(len(rhomboids_tiling_vertexes_set)):
#     plot_3d_convex_hull(np.array(rhomboids_tiling_vertexes_set[i]),labels_set[i],edges_coord_set)
print(rhomboids_tiling_vertexes_set[0])

plot_rhomboidtiling(rhomboids_tiling_vertexes_set,np.array(all_vertexes),all_labels_set,edges_coord_set,edges_Del,triangles_Del)
plot_one_rhomboid_with_rhomboidtiling(rhomboids_tiling_vertexes_set[0],rhomboids_tiling_vertexes_set,np.array(all_vertexes),all_labels_set,edges_coord_set,edges_Del,triangles_Del)
#fig.show()

