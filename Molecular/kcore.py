'''import dgl
import torch
import networkx as nx
import math


def K_core(graph):
    nx_graph = graph.to_networkx().to_undirected()
    G = nx.Graph(nx_graph)
    #degree
    Deg = {}
    for node in G.nodes():
        Deg[node] = G.degree(node)
    # K_Core
    List_all ={}
    coreness = {}
    Node_list = list(G.nodes())
    k = 0
    while len(Node_list) != 0:   
        Flag = True
        List_k = []
        while Flag == True:
            Remove_list = []
            for v in Node_list:
                if 0<= Deg[v] <= k:  
                    List_k.append(v)
                    Remove_list.append(v)
                    Node_list.remove(v)
                    coreness[v] = k
            if len(Remove_list) != 0:
                if k > 0:
                    for v in Remove_list:     
                        for u in list(G[v]):
                            Deg[u] = Deg[u] -1
                Flag = True
            else:
                Flag = False
        List_all[k] = List_k
        k = k + 1

    core_val = [0] * graph.number_of_nodes()
    for node in coreness.keys():
        val = coreness[node]
        num_val = len(List_all[val])
        core_val[node] = [math.log(val)/num_val] if val > 1 else [0.5]

    graph.ndata['kcore'] = torch.tensor(core_val, dtype=torch.float32)
    return graph



if __name__ == '__main__':
    # 创建一个 DGL 图
    g = dgl.DGLGraph()

    g.add_nodes(6)
    src_list = [0, 0, 1, 2, 2, 2]
    dst_list = [1, 2, 3, 4, 3, 4]
    g.add_edges(src_list, dst_list)
    G = K_core(g)
    print(G.ndata['kcore'])
'''

import torch
import torch_geometric
from torch_geometric.data import Data
import networkx as nx
import math

def K_core(graph):
    nx_graph = torch_geometric.utils.to_networkx(graph, to_undirected=True)
    G = nx.Graph(nx_graph)

    #degree
    Deg = {}
    for node in G.nodes():
        Deg[node] = G.degree(node)
    # K_Core
    List_all ={}
    coreness = {}
    Node_list = list(G.nodes())
    k = 0
    while len(Node_list) != 0:   
        Flag = True
        List_k = []
        while Flag == True:
            Remove_list = []
            for v in Node_list:
                if 0<= Deg[v] <= k:  
                    List_k.append(v)
                    Remove_list.append(v)
                    Node_list.remove(v)
                    coreness[v] = k
            if len(Remove_list) != 0:
                if k > 0:
                    for v in Remove_list:     
                        for u in list(G[v]):
                            Deg[u] = Deg[u] -1
                Flag = True
            else:
                Flag = False
        List_all[k] = List_k
        k = k + 1

    core_val = [0] * graph.num_nodes
    for node in coreness.keys():
        val = coreness[node]
        num_val = len(List_all[val])
        core_val[node] = [math.log(val)/num_val] if val > 1 else [0.5/num_val]


    # Add k-core values as a new feature in the graph
    graph.kcore = torch.tensor(core_val, dtype=torch.float32).view(-1, 1)
    return graph

# Example usage
if __name__ == '__main__':
    # Create a graph using PyTorch Geometric
    edge_index = torch.tensor([[0, 0, 1, 2, 2, 2], [1, 2, 3, 4, 3, 4]], dtype=torch.long)
    x = torch.randn(6, 3)  # Example node features
    data = Data(x=x, edge_index=edge_index)

    # Compute k-core and add it as a new feature
    result = K_core(data)
    print("Original Features:\n", result.x)
    print("K-core Features:\n", result.kcore.tolist())
