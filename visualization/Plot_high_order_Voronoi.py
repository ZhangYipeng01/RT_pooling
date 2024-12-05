# -*- coding: utf-8 -*-
import re
import numpy as np
import plotly.graph_objects as go

def circumcenter(a, b, c):
    """给定三维空间的四个点，求外接球球心"""
    A = np.vstack((b - a, c - a))
    bvec = np.array([np.dot(b-a, b-a),
                     np.dot(c-a, c-a)])


    circ_center = np.linalg.solve(2 * A, bvec) + a
    return circ_center

def vertex_get_coords(vertex):
    coord = np.array([0.0, 0.0])
    for index in vertex:
        coord = np.array(coord) + np.array(coords[index])
    return coord.tolist()

def get_on_S_indices(simplex):
    sets = [set(sublist) for sublist in simplex]
    union_set = set.union(*sets)
    intersection_set = set.intersection(*sets)
    on_s_set = union_set - intersection_set
    on_s_list = list(on_s_set)
    return on_s_list

def get_in_S_indices(simplex):
    sets = [set(sublist) for sublist in simplex]
    intersection_set = set.intersection(*sets)
    in_s_list = list(intersection_set)
    return in_s_list


def plot_Voronoi(filename, input_max_order):
    global coords
    coords = np.loadtxt(filename + '.txt')

    dslice = []
    with open(filename+'_fslices.txt', 'r') as file:
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

    r_fslices_set = []
    with open(filename + '_fslices.txt', 'r') as file:
        for line in file:
            line = line.strip()

            if line.startswith('Slice'):
                current_slice = int(line.split(' ')[1].replace(':', ''))
            elif line:
                tuple_data = eval(line)
                r_fslices_set.append([tuple_data[0], current_slice, tuple_data[1]])

    max_order = r_fslices_set[-1][1]
    if max_order < input_max_order:
        print('The input max order is bigger than what we have in the fslice file')
        max_order = input_max_order
    else:
        max_order = input_max_order

    vertices = []
    edges = []
    triangles = []
    for i in range(max_order):
        vertices.append([])
        edges.append([])
        triangles.append([])
    for line in r_fslices_set:
        simplex = line[0]
        order = int(line[1])
        if order <= max_order:
            if len(simplex) == 1:
                vertices[order-1].append(simplex)
            if len(simplex) == 2:
                edges[order-1].append(simplex)
            if len(simplex) >= 3:
                triangles[order-1].append(simplex)

    mins = coords.min(axis=0)  # [x_min, y_min, z_min]
    maxs = coords.max(axis=0)  # [x_max, y_max, z_max]
    ranges = maxs - mins
    expanded_mins = mins - 0.2 * ranges
    expanded_maxs = maxs + 0.2 * ranges

    fig = go.Figure()
    for idx in range(len(coords)):
        v = coords[idx]
        x, y = [float(v[0])], [float(v[1])]  # 获取所有点的 x 和 y 坐标
        fig.add_trace(go.Scatter(
            x=x, y=y, mode='markers+text',  # 同时显示点和标签
            marker=dict(size=16, color='black'),  # 点的大小为 8（边的粗细为 2 的两倍）
            #text=str(idx),  # 设置点的标签
            textposition="top center",  # 标签位置
            textfont=dict(size=12, color="black")  # 标签字体样式
        ))

    fig.update_layout(
        xaxis=dict(
            range=[expanded_mins[0], expanded_maxs[0]],
            visible=False,  # 隐藏 X 轴
            scaleanchor="y"  # 锁定 X 和 Y 轴比例一致
        ),
        yaxis=dict(
            range=[expanded_mins[1], expanded_maxs[1]],
            visible=False  # 隐藏 Y 轴
        ),
        plot_bgcolor="white",  # 设置背景为纯白
        paper_bgcolor="white",  # 设置图表外部为纯白
        showlegend=False  # 隐藏图例
    )

    fig.show()


    for i in range(max_order):
        fig = go.Figure()
        slice = dslice[i]
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

        # 绘制三角形
        for triangle in t:
            x, y = zip(*triangle)  # 解包顶点坐标
            x += (x[0],)  # 闭合三角形
            y += (y[0],)
            fig.add_trace(go.Scatter(
                x=x, y=y, fill='toself', mode='lines',
                line=dict(color='red'), fillcolor='rgba(179, 230, 255, 0.9)', opacity=0.05
            ))

        # 绘制边
        for edge in e:
            x, y = zip(*edge)  # 解包顶点坐标
            fig.add_trace(go.Scatter(
                x=x, y=y, mode='lines', line=dict(color='red', width=4), opacity=0.1
            ))

        # 绘制点
        for j in range(len(v)):
            x, y = zip(*v[j])  # 获取所有点的 x 和 y 坐标
            fig.add_trace(go.Scatter(
                x=x, y=y, mode='markers+text',  # 同时显示点和标签
                marker=dict(size=16, color='red'),  # 点的大小为 8（边的粗细为 2 的两倍）
                # text=v_labels[j],  # 设置点的标签
                textposition="top center",  # 标签位置
                textfont=dict(size=12, color="black"),  # 标签字体样式
                opacity=0.1
            ))

        # for idx in range(len(coords)):
        #     v = coords[idx]
        #     x, y = [float(v[0])], [float(v[1])]  # 获取所有点的 x 和 y 坐标
        #     fig.add_trace(go.Scatter(
        #         x=x, y=y, mode='markers+text',  # 同时显示点和标签
        #         marker=dict(size=16, color='black'),  # 点的大小为 8（边的粗细为 2 的两倍）
        #         #text=str(idx),  # 设置点的标签
        #         textposition="top center",  # 标签位置
        #         textfont=dict(size=12, color="black")  # 标签字体样式
        #     ))

        dual_coords = dict()
        for triangle in triangles[i]:
            on_s_list = get_on_S_indices(triangle)
            if len(on_s_list) == 3:
                v1 = coords[on_s_list[0]]
                v2 = coords[on_s_list[1]]
                v3 = coords[on_s_list[2]]
                dual_coord = circumcenter(v1, v2, v3)
                dual_coords[str(triangle)] = dual_coord
            else:
                print('coords are not in general position')

        dual_edges = dict()
        #is_triangle_boundary = dict()
        for idx in range(len(edges[i])):
        #for idx in [1]:
            edge = edges[i][idx]
            print('edge',edge)
            on_s_list = get_on_S_indices(edge)
            if len(on_s_list) == 2:
                #is_triangle_boundary[str(triangle)] = False
                edge_set = {tuple(e) for e in edge}
                nodes_in_edge = []
                for triangle in triangles[i]:
                    triangle_set = {tuple(t) for t in triangle}
                    if edge_set.issubset(triangle_set):
                        upper_triangle = triangle
                        nodes_in_edge.append(dual_coords[str(triangle)])
                if len(nodes_in_edge) == 2:
                    dual_edges[str(edge)] = nodes_in_edge

                if len(nodes_in_edge) == 1:
                    centroid = (coords[on_s_list[0]] + coords[on_s_list[1]])/2
                    if np.linalg.norm(centroid - nodes_in_edge[0]) != 0:
                        normal = (centroid - nodes_in_edge[0])/np.linalg.norm(centroid - nodes_in_edge[0])
                    else:
                        link = coords[on_s_list[0]] - coords[on_s_list[1]]
                        normal = np.array([-link[1],link[0]]) / np.linalg.norm(np.array([-link[1],link[0]]))
        # 10000 below is a big number to make sure nodes_in_edge[0] + 10000*normal is outside of given boundary,
        # please change it to a suitable number if it's not large enough
                    inside_points = get_in_S_indices(edge)
                    if (len(inside_points) == 0):
                        nodes_in_edge.append(nodes_in_edge[0] + 10000 * normal)
                    else:
                        if (np.linalg.norm(nodes_in_edge[0] + 10000 * normal - coords[on_s_list[0]]) >= np.linalg.norm(nodes_in_edge[0] + 10000 * normal - coords[inside_points[0]])):
                            nodes_in_edge.append(nodes_in_edge[0] + 10000 * normal)
                        else:
                            nodes_in_edge.append(nodes_in_edge[0] - 10000 * normal)
                    dual_edges[str(edge)] = nodes_in_edge

                print('nodes_in_edge',nodes_in_edge)
                x, y = zip(*nodes_in_edge)
                fig.add_trace(go.Scatter(
                    x=x, y=y, mode='lines', line=dict(color='black', width=5)  # 边的粗细为 2
                ))
            else:
                print('Something is wrong about edge:', edge)

        fig.update_layout(
            xaxis=dict(
                range=[expanded_mins[0], expanded_maxs[0]],
                visible=False,  # 隐藏 X 轴
                scaleanchor="y"  # 锁定 X 和 Y 轴比例一致
            ),
            yaxis=dict(
                range=[expanded_mins[1], expanded_maxs[1]],
                visible=False  # 隐藏 Y 轴
            ),
            plot_bgcolor="white",  # 设置背景为纯白
            paper_bgcolor="white",  # 设置图表外部为纯白
            showlegend=False  # 隐藏图例
        )

        fig.show()

plot_Voronoi('4points',3)