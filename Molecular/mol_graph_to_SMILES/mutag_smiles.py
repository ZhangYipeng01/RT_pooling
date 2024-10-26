import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem import rdMolDescriptors

# 加载文件数据
edges = np.loadtxt("./data/MUTAG/MUTAG_A.txt", delimiter=",", dtype=int)
edge_labels = np.loadtxt("./data/MUTAG/MUTAG_edge_labels.txt", dtype=int)
node_labels = np.loadtxt("./data/MUTAG/MUTAG_node_labels.txt", dtype=int)
graph_indicator = np.loadtxt("./data/MUTAG/MUTAG_graph_indicator.txt", dtype=int)
graph_labels = np.loadtxt("./data/MUTAG/MUTAG_graph_labels.txt", dtype=int)

# 映射节点标签到原子类型
atom_map = {0: "C", 1: "N", 2: "O", 3: "F", 4: "I", 5: "Cl", 6: "Br"}  # 从0开始
# 映射边标签到键类型
bond_map = {0: rdchem.BondType.AROMATIC, 1: rdchem.BondType.SINGLE, 
            2: rdchem.BondType.DOUBLE, 3: rdchem.BondType.TRIPLE}

# 根据图标识分割图
num_graphs = max(graph_indicator)
graphs_nodes_tmp = {i: [] for i in range(1, num_graphs + 1)}
graph_nodes_label = {i: [] for i in range(1, num_graphs + 1)}


for idx, graph_id in enumerate(graph_indicator, start=1):
    graphs_nodes_tmp[graph_id].append(idx)
    graph_nodes_label[graph_id].append(node_labels[idx-1])  

graphs_nodes= {i: [] for i in range(1, num_graphs + 1)}

ini = 0
for i in range(1, num_graphs + 1):
    graph_nodes_i = graphs_nodes_tmp[i]
    for node in graph_nodes_i:
        node = node - ini
        graphs_nodes[i].append(node)
    ini += len(graphs_nodes[i])


graph_edges = {i: [] for i in range(1, num_graphs + 1)}
graph_edges_bond = {i: [] for i in range(1, num_graphs + 1)}
ini = 0
for i in range(1,num_graphs+1):
    min_max = []
    geaph_edges_i = len(graphs_nodes[i])

    graph_edges_i = []
    graph_edges_bond_i = []

    for j in range(len(edges)):
        u, v = edges[j]
        bond_type = edge_labels[j]

        if ini+1 <= u <= ini + geaph_edges_i and ini+1 <= v <= ini + geaph_edges_i:
            star = u - ini
            end = v - ini
            graph_edges_i.append([star,end])
            graph_edges_bond_i.append(bond_type)
    
    graph_edges[i] = graph_edges_i
    graph_edges_bond[i] = graph_edges_bond_i
    ini = ini + geaph_edges_i



# 构建每个图的分子并生成SMILES及其标签
smiles_list = []
labels = []
print('The number of the graphs:', len(graph_edges))
for graph_id in range(1, num_graphs + 1):
    
    if graph_id == 2:
        print(graph_edges[graph_id])
        print(graphs_nodes[graph_id])


    edges = graph_edges[graph_id]
    atom_types = graph_nodes_label[graph_id]
    bond_type  = graph_edges_bond[graph_id]
    max_atom_index = max(max(pair) for pair in edges)
    

    mol = Chem.RWMol()

    for i in range(max_atom_index):
        tmp = atom_map[atom_types[i]]
        atom = Chem.Atom(tmp)
        mol.AddAtom(atom)

    added_bonds = set()
    for i in range(len(edges)):
        start, end = edges[i]
        edge_bond = bond_type[i] 
        if (start, end) not in added_bonds and (end, start) not in added_bonds:
            
            if edge_bond == 3:
                mol.AddBond(int(start) - 1, int(end) - 1, bond_map[edge_bond])
            else:
                mol.AddBond(int(start) - 1, int(end) - 1, rdchem.BondType.SINGLE)  # 先添加为单键
            added_bonds.add((start, end))
                

    # 使用SanitizeMol前进行环检测
    Chem.SanitizeMol(mol)  # 确保分子化学结构正确
    ssr = Chem.GetSymmSSSR(mol)  # 获取环信息

    # 将环中的原子标记为芳香性
    for ring in ssr:
        if len(ring) == 6:  # 假设寻找六元环
            for idx in ring:
                atom = mol.GetAtomWithIdx(idx)
                atom.SetIsAromatic(True)
                for n in atom.GetNeighbors():
                    if n.GetIdx() in ring:
                        mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).SetBondType(rdchem.BondType.AROMATIC)

    smiles = Chem.MolToSmiles(mol)
    # 再次清理分子结构
    Chem.SanitizeMol(mol)  
    # 生成SMILES并获取标签
    try:
        smiles = Chem.MolToSmiles(mol)
        smiles_list.append(smiles)
        labels.append(graph_labels[graph_id - 1])
        if graph_id == 1:
            print(smiles)  

    except Exception as e:
        print(f"Failed to generate SMILES for graph {graph_id}: {e}")

# 创建DataFrame并保存为CSV
df = pd.DataFrame({"SMILES": smiles_list, "Label": labels})
df.to_csv("./results/MUTAG_smiles_labels.csv", index=False)
print("SMILES和标签已成功保存到MUTAG_smiles_labels.csv")