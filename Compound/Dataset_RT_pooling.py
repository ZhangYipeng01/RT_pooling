from config import RT_order_list
import argparse
import re
import numpy as np
from torch_geometric.datasets import TUDataset
import torch
import os

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

def rhomboid_reader(filename):
    r_rhomboids_set = []
    with open(filename+'_rhomboids.txt', 'r') as file:
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
            r_rhomboids_set.append(temp_result)

    r_fslices_set = []
    current_slice = 0
    max_dim = 0
    with open(filename+'_fslices.txt', 'r') as file:
        for line in file:
            line = line.strip()  # 去除两端的空格和换行符

            # 检查是否是Slice行，如果是，更新当前Slice编号
            if line.startswith('Slice'):
                current_slice = int(line.split(' ')[1].replace(':', ''))
            elif line:  # 忽略空行
                # 解析当前行的数据
                tuple_data = eval(line)  # 将字符串转换成元组
                # 添加到结果列表，包括Slice编号
                r_fslices_set.append([tuple_data[0], current_slice, tuple_data[1]])

    return [r_rhomboids_set,r_fslices_set]

def get_incidence_matrices(filename,order_list):
    result = rhomboid_reader(filename)
    top_rhomboids_set = result[0]
    num_top_rhomboids = len(top_rhomboids_set)
    fslices_set = result[1]
    vertex_list = {}
    incidence_matrix_list = {}
    for order in order_list:
        vertex_list[order] = []

    for simplex in fslices_set:
        k = simplex[1]
        try:
            if len(simplex[0]) == 1:
                vertex_list[k].append(simplex[0][0])
        except KeyError:
            pass
    vertex_list[1] = sorted(vertex_list[1], key=lambda x: x[0])

    for order in order_list:
        num_vertices = len(vertex_list[order])
        B = np.zeros((num_top_rhomboids, num_vertices))
        for i in range(num_vertices):
            for j in range(num_top_rhomboids):
                set_vertex = set(vertex_list[order][i])
                set_in_S = set(top_rhomboids_set[j][0])
                set_rhomboid = set(top_rhomboids_set[j][1])
                if (set_in_S.issubset(set_vertex)) and (set_vertex.issubset(set_rhomboid|set_in_S)):
                    B[j,i] = 1
        incidence_matrix_list[order] = B
    return incidence_matrix_list

def get_matrices_for_pooling(filename, order_list):
    total_filename = filename

    fslices_set = rhomboid_reader(total_filename)[1]
    vertex_list = {}
    for order in order_list:
        vertex_list[order] = []
    for simplex in fslices_set:
        k = simplex[1]
        try:
            if len(simplex[0]) == 1:
                vertex_list[k].append(simplex[0][0])
        except KeyError:
            pass
    vertex_list[1] = sorted(vertex_list[1], key=lambda x: x[0])

    incidence_matrices_dict = get_incidence_matrices(total_filename, order_list)

    RT_matrix_list = []
    for i in range(len(order_list)-1):
        A = np.dot(incidence_matrices_dict[order_list[i+1]].T,incidence_matrices_dict[order_list[i]])
        input_vertices = vertex_list[order_list[i]]
        output_vertices = vertex_list[order_list[i+1]]
        for x in range(len(input_vertices)):
            for y in range(len(output_vertices)):
                if not set(input_vertices[x]).issubset(set(output_vertices[y])):
                    A[y,x] = 0

        row_sums = A.sum(axis=1)
        normalized_A = np.zeros_like(A, dtype=float)
        for i in range(A.shape[0]):
            if row_sums[i] != 0:
                normalized_A[i] = A[i] / row_sums[i]
            else:
                normalized_A[i] = A[i]  # 保持全零行不变

        RT_matrix_list.append(normalized_A)

    #RT_matrix = RT_matrix_list[0]
    #for i in range(len(RT_matrix_list)-1):
        #RT_matrix = np.dot(RT_matrix_list[i+1],RT_matrix)

    return RT_matrix_list
    
def get_delaunay_edge_indexes(filename, order_list):
    total_filename = filename

    fslices_set = rhomboid_reader(total_filename)[1]
    vertex_list = {}
    for order in order_list:
        vertex_list[order] = []
    for simplex in fslices_set:
        k = simplex[1]
        try:
            if len(simplex[0]) == 1:
                vertex_list[k].append(simplex[0][0])
        except KeyError:
            pass
    vertex_list[1] = sorted(vertex_list[1], key=lambda x: x[0])

    incidence_matrices_dict = get_incidence_matrices(total_filename, order_list)
    
    edge_index_list = []
    for i in range(len(order_list)):
        A = np.dot(incidence_matrices_dict[order_list[i]].T,incidence_matrices_dict[order_list[i]])
        rows, cols = np.nonzero(A)
        edge_index = np.vstack((rows, cols))
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_index_list.append(edge_index)
        
    return edge_index_list

def get_gen_edge_indexes(dataset, order_list):
    ori_graphs = TUDataset(root='/tmp/' + dataset, name = dataset)
    data_gen_edge_indexes = {}
    for i, graph in enumerate(ori_graphs):
        ori_edge_index = graph.edge_index

        filename = './data/'+dataset+'/filtrations/'+str(i)
        fslices_set = rhomboid_reader(filename)[1]
        vertex_list = {}
        for order in order_list:
            vertex_list[order] = []
        for simplex in fslices_set:
            k = simplex[1]
            try:
                if len(simplex[0]) == 1:
                    vertex_list[k].append(simplex[0][0])
            except KeyError:
                pass
        vertex_list[1] = sorted(vertex_list[1], key=lambda x: x[0])

        gen_edge_indexes = []
        for order in order_list:
            gen_edge_index_set = set()  # 使用集合来存储边，避免重复

            for edge in ori_edge_index.T:
                x, y = edge.tolist()  # 获取边的两个顶点 i 和 j
                x_indices = [idx for idx, element in enumerate(vertex_list[order]) if x in element]
                y_indices = [idx for idx, element in enumerate(vertex_list[order]) if y in element]

                for x_idx in x_indices:
                    for y_idx in y_indices:
                        # 添加边和其反向边到集合中，去除重复
                        gen_edge_index_set.add((x_idx, y_idx))
                        gen_edge_index_set.add((y_idx, x_idx))

            # 将集合转换为列表
            gen_edge_index = list(gen_edge_index_set)
            gen_edge_index = torch.tensor(gen_edge_index, dtype=torch.long).T

            gen_edge_indexes.append(gen_edge_index)

        data_gen_edge_indexes[i] = gen_edge_indexes

    return data_gen_edge_indexes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='RTpooling for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="COX2",
                        help='name of dataset (default: COX2)')
    args = parser.parse_args()

    pre = './data/'+args.dataset+'/'
    num_graph = len([f for f in os.listdir(pre+'coordinates/') if f.endswith('.txt')])
    print(num_graph)
    
    data_RT_matrix = {}
    data_DL_edge_indexes = {}
    for i in range(num_graph):
        try:
           RT_pooling_matrix = get_matrices_for_pooling(pre + 'filtrations/'+str(i), RT_order_list)
           data_RT_matrix[i] = RT_pooling_matrix
           Delaunay_edge_indexes = get_delaunay_edge_indexes(pre + 'filtrations/'+str(i), RT_order_list)
           data_DL_edge_indexes[i] = Delaunay_edge_indexes
        except Exception as e:
           print(f"An error occurred at i = {i}: {e}")

    data_gen_edge_indexes = get_gen_edge_indexes(args.dataset, RT_order_list)

    torch.save(data_RT_matrix, pre + '/data_RT_matrices.pt')
    torch.save(data_DL_edge_indexes, pre + '/Delaunay_edge_indexes.pt')
    torch.save(data_gen_edge_indexes, pre + '/Gen_edge_indexes.pt')

    
