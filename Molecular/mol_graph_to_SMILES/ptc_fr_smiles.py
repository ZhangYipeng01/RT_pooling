import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdchem
from check_mol_C_keku import adjust_valence_exception_C
from check_mol_N import adjust_nitrogen_valence_N


dataset_name = 'PTC_FR'
# 加载文件数据
edges = np.loadtxt(f"./data/{dataset_name}/{dataset_name}_A.txt", delimiter=",", dtype=int)  # 读取边信息
edge_labels = np.loadtxt(f"./data/{dataset_name}/{dataset_name}_edge_labels.txt", dtype=int)  # 读取键类型信息
node_labels = np.loadtxt(f"./data/{dataset_name}/{dataset_name}_node_labels.txt", dtype=int)  # 读取节点标签
graph_indicator = np.loadtxt(f"./data/{dataset_name}/{dataset_name}_graph_indicator.txt", dtype=int)  # 图标识
graph_labels = np.loadtxt(f"./data/{dataset_name}/{dataset_name}_graph_labels.txt", dtype=int)  # 图标签

# 映射节点标签到原子类型
atom_map = {
    0: "In", 1: "P", 2: "O", 3: "N", 4: "Na", 5: "C", 6: "Cl", 7: "S",
    8: "Br", 9: "F", 10: "As", 11: "K", 12: "Cu", 13: "Zn", 14: "I",
    15: "Sn", 16: "Pb", 17: "Te", 18: "Ca"
}
# 映射边标签到键类型
bond_map = {
    0: rdchem.BondType.TRIPLE, 1: rdchem.BondType.SINGLE,
    2: rdchem.BondType.DOUBLE, 3: rdchem.BondType.AROMATIC
}

# 根据图标识分割图
num_graphs = max(graph_indicator)
graphs_nodes_tmp = {i: [] for i in range(1, num_graphs + 1)}
graph_nodes_label = {i: [] for i in range(1, num_graphs + 1)}

for idx, graph_id in enumerate(graph_indicator, start=1):
    graphs_nodes_tmp[graph_id].append(idx)
    graph_nodes_label[graph_id].append(node_labels[idx - 1])

graphs_nodes = {i: [] for i in range(1, num_graphs + 1)}

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
for i in range(1, num_graphs + 1):
    min_max = []
    geaph_edges_i = len(graphs_nodes[i])

    graph_edges_i = []
    graph_edges_bond_i = []

    for j in range(len(edges)):
        u, v = edges[j]
        bond_type = edge_labels[j]

        if ini + 1 <= u <= ini + geaph_edges_i and ini + 1 <= v <= ini + geaph_edges_i:
            star = u - ini
            end = v - ini
            graph_edges_i.append([star, end])
            graph_edges_bond_i.append(bond_type)

    graph_edges[i] = graph_edges_i
    graph_edges_bond[i] = graph_edges_bond_i
    ini = ini + geaph_edges_i

# 构建每个图的分子并生成SMILES及其标签
smiles_list = []
labels = []
print('The number of the graphs:', len(graph_edges))
for graph_id in range(1, num_graphs + 1):

    # print(graph_id)

    edges = graph_edges[graph_id]
    atom_types = graph_nodes_label[graph_id]
    bond_type = graph_edges_bond[graph_id]
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

            '''
            if edge_bond in [0,2]:
                mol.AddBond(int(start) - 1, int(end) - 1, bond_map[edge_bond])
            else:
                mol.AddBond(int(start) - 1, int(end) - 1, rdchem.BondType.SINGLE)  # 先添加为单键
            '''
            if len(edges) == 2:
                mol.AddBond(int(start) - 1, int(end) - 1, bond_map[edge_bond])
            else:
                mol.AddBond(int(start) - 1, int(end) - 1, rdchem.BondType.SINGLE)

            # mol.AddBond(int(start) - 1, int(end) - 1, rdchem.BondType.SINGLE)
            added_bonds.add((start, end))

    smiles = Chem.MolToSmiles(mol)
    # print(smiles)
    # 清理分子结构
    mol = adjust_valence_exception_C(mol)
    if mol is None:  # 检查分子对象是否成功创建
        print('!!!!!!!!!')
        print(edges)
        print(atom_types)
        print(bond_type)
        continue

    else:

        mol = adjust_nitrogen_valence_N(mol)

        # 使用SanitizeMol前进行环检测
        try:
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

            # 生成SMILES并获取标签
            mol = Chem.AddHs(mol)
            smiles = Chem.MolToSmiles(mol)
            try:
                valid_mol = Chem.MolFromSmiles(smiles)
                valid_mol = Chem.AddHs(valid_mol)
            except Exception as e:
                print(e)
                print('Error SMILES:', smiles)

            smiles_list.append(smiles)
            labels.append(graph_labels[graph_id - 1])


        except Chem.rdchem.MolSanitizeException as e:
            print(f"Failed to sanitize molecule from: {e}")
            continue  # 处理下一个分子，跳过当前的错误处理

# 创建DataFrame并保存为CSV
df = pd.DataFrame({"smiles": smiles_list, "label": labels})
df.to_csv(f"./results/{dataset_name}.csv", index=False)
print(f"SMILES和标签已成功保存到{dataset_name}.csv")