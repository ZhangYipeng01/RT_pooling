# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import pandas as pd
import networkx as nx
 
 
from rdkit import Chem
from rdkit.Chem import AllChem
from jarvis.core.specie import chem_data, get_node_attributes
from torch_geometric.data import Data
 
def normalize_columns_01(input_tensor):
    # Assuming input_tensor is a 2D tensor (matrix)
    min_vals, _ = input_tensor.min(dim=0)
    max_vals, _ = input_tensor.max(dim=0)
 
    # Identify columns where max and min are not equal
    non_zero_mask = max_vals != min_vals
 
    # Avoid division by zero
    normalized_tensor = input_tensor.clone()
    normalized_tensor[:, non_zero_mask] = (input_tensor[:, non_zero_mask] - min_vals[non_zero_mask]) / (max_vals[non_zero_mask] - min_vals[non_zero_mask] + 1e-10)
 
    return normalized_tensor
 
 
 
def calculate_dis(A,B):
    AB = B - A
    dis = np.linalg.norm(AB)
    return dis
 
 
 
 
 
def bond_length_approximation(bond_type):
    bond_length_dict = {"SINGLE": 1.0, "DOUBLE": 1.4, "TRIPLE": 1.8, "AROMATIC": 1.5}
    return bond_length_dict.get(bond_type, 1.0)
 
def encode_bond_14(bond):
    #7+4+2+2+6 = 21
    bond_dir = [0] * 7
    bond_dir[bond.GetBondDir()] = 1
    
    bond_type = [0] * 4
    bond_type[int(bond.GetBondTypeAsDouble()) - 1] = 1
    
    bond_length = bond_length_approximation(bond.GetBondType())
    
    in_ring = [0, 0]
    in_ring[int(bond.IsInRing())] = 1
    
    non_bond_feature = [0]*6
 
    edge_encode = bond_dir + bond_type + [bond_length,bond_length**2] + in_ring + non_bond_feature
 
    return edge_encode
 
 
 
def non_bonded(charge_list,i,j,dis):
    charge_list = [float(charge) for charge in charge_list]
    q_i = [charge_list[i]]
    q_j = [charge_list[j]]
    q_ij = [charge_list[i]*charge_list[j]]
    dis_1 = [1/dis]
    dis_2 = [1/(dis**6)]
    dis_3 = [1/(dis**12)]
 
    return q_i + q_j + q_ij + dis_1 + dis_2 +dis_3
 
 
def combined_force_field(mol):
    max_tries = 10
    num_tries = 0

    for i in range(max_tries):
        num_tries += 1
        try:
            if mmff_force_field(mol):
                return True  
            else:
                AllChem.EmbedMolecule(mol, randomSeed=10, useRandomCoords=True)            
                rdForceFieldHelpers.MMFFOptimizeMolecule(mol, maxIters=2147483647)

        except BaseException as be:
            if num_tries < max_tries:
                continue 
            else:
                print(f"Error processing molecule: {be}")
                return False

    return False
 
 
def mmff_force_field(mol):
    try:
        # 尝试嵌入分子
        AllChem.EmbedMultipleConfs(mol, numConfs=10, randomSeed=42)
        # 创建 MMFF 力场
        AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
        return True
    except ValueError:
        # 如果捕获到ValueError异常，则无法嵌入分子，返回False
        return False
 
 
def uff_force_field(mol):
    try:
        # 尝试嵌入分子
        AllChem.EmbedMolecule(mol)
        # 创建 MMFF 力场
        AllChem.UFFGetMoleculeForceField(mol)
        return True
    except ValueError:
        # 如果捕获到ValueError异常，则无法嵌入分子，返回False
        return False
    
def random_force_field(mol):
    try:
        AllChem.EmbedMolecule(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=10, randomSeed=42)
        return True
    except ValueError:
        # 如果捕获到ValueError异常，则无法嵌入分子，返回False
        return False
 
 
def check_common_elements(list1, list2, element1, element2):
    if len(list1) != len(list2):
        return False  # 如果列表长度不相同，直接返回 False
    
    for i in range(len(list1)):
        if list1[i] == element1 and list2[i] == element2:
            return True  # 如果找到一对匹配的元素，返回 True
    
    return False  # 如果没有找到匹配的元素，返回 False
 
def atom_to_graph(smiles,encoder_atom):
    
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)  # 加氢
    sps_features = []
    coor = []
    edge_id = []
    atom_charges = []
    
    smiles_with_hydrogens = Chem.MolToSmiles(mol)
 
    tmp = []
    for num in smiles_with_hydrogens:
        if num not in ['[',']','(',')']:
            tmp.append(num)
 
    sm = {}
    for atom in mol.GetAtoms():
        atom_index = atom.GetIdx()
        sm[atom_index] =  atom.GetSymbol()
 
    Num_toms = len(tmp)
    if Num_toms > 1500:
        g = False
 
    else:
        if combined_force_field(mol) == True:
            num_conformers = mol.GetNumConformers()
            
            if num_conformers > 0:
                AllChem.ComputeGasteigerCharges(mol)
                for ii, s in enumerate(mol.GetAtoms()):
 
                    per_atom_feat = []
                            
                    feat = list(get_node_attributes(s.GetSymbol(), atom_features=encoder_atom))
                    per_atom_feat.extend(feat)
 
                    sps_features.append(per_atom_feat )
                        
                    
                    # 获取原子坐标信息
                    pos = mol.GetConformer().GetAtomPosition(ii)
                    coor.append([pos.x, pos.y, pos.z])
 
                    # 获取电荷并存储
                    charge = s.GetProp("_GasteigerCharge")
                    atom_charges.append(charge)
 
                edge_features = []
                edge_index = []
                for bond in mol.GetBonds():
                    src = bond.GetBeginAtomIdx()
                    dst = bond.GetEndAtomIdx()
                    edge_index += [[src, dst], [dst, src]]
 
                    per_bond_feat = []
                    per_bond_feat.extend(encode_bond_14(bond))
 
                    edge_features.append(per_bond_feat)
                    edge_features.append(per_bond_feat)
                    
                pos_tensor = torch.tensor(coor, dtype=torch.float32)
                edge_feats = torch.tensor(edge_features, dtype=torch.float32)
                node_feats = torch.tensor(sps_features,dtype=torch.float32)
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
                g = Data(x = node_feats, edge_index = edge_index, pos = pos_tensor)
            
            else:
                g = False
        else:
            g = False
    return g


def atom_to_graph_TUDataset(smiles, atom_map):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    sps_features = []
    coor = []
    edge_index = []
    atom_charges = []
    
    smiles_with_hydrogens = Chem.MolToSmiles(mol)
 
    tmp = []
    for num in smiles_with_hydrogens:
        if num not in ['[',']','(',')']:
            tmp.append(num)
 
    sm = {}
    for atom in mol.GetAtoms():
        atom_index = atom.GetIdx()
        sm[atom_index] = atom.GetSymbol()
 
    Num_atoms = len(tmp)
    if Num_atoms > 1500:
        g = False
    else:
        if mmff_force_field(mol): 
            num_conformers = mol.GetNumConformers()
            
            if num_conformers > 0:
                AllChem.ComputeGasteigerCharges(mol)
                
                for ii, s in enumerate(mol.GetAtoms()):
                    atom_symbol = s.GetSymbol()
                    
                    # 跳过氢原子
                    if atom_symbol == "H":
                        continue
                    
                    # 根据 atom_map 生成 one-hot 编码
                    one_hot_feat = [0] * len(atom_map)
                    if atom_symbol in atom_map.values():
                        atom_type_index = list(atom_map.values()).index(atom_symbol)
                        one_hot_feat[atom_type_index] = 1
                    
                    # 将 one-hot 特征加入节点特征列表
                    sps_features.append(one_hot_feat)
                    
                    # 获取非氢原子的坐标信息
                    pos = mol.GetConformer().GetAtomPosition(ii)
                    coor.append([pos.x, pos.y, pos.z])
 
                    # 获取电荷并存储
                    charge = s.GetProp("_GasteigerCharge")
                    atom_charges.append(charge)
 
                edge_features = []
                for bond in mol.GetBonds():
                    src = bond.GetBeginAtomIdx()
                    dst = bond.GetEndAtomIdx()

                    # 跳过与氢原子相关的边
                    if mol.GetAtomWithIdx(src).GetSymbol() == "H" or mol.GetAtomWithIdx(dst).GetSymbol() == "H":
                        continue

                    edge_index += [[src, dst], [dst, src]]
 
                    # 边的特征（如有需要，可以在这里编码边的特征）
                    per_bond_feat = []
                    per_bond_feat.extend(encode_bond_14(bond))  # 假设你已经定义了这个函数
 
                    edge_features.append(per_bond_feat)
                    edge_features.append(per_bond_feat)
                    
                # 转换为 PyTorch 的 tensor
                pos_tensor = torch.tensor(coor, dtype=torch.float32)
                edge_feats = torch.tensor(edge_features, dtype=torch.float32)
                node_feats = torch.tensor(sps_features, dtype=torch.float32)
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
                # 构建图对象
                g = Data(x=node_feats, edge_index=edge_index, pos=pos_tensor)
            
            else:
                g = False
        else:
            g = False
    return g


def Similes_Graph(Smile, encoder_atom):
    #generate graph
    # 创建一个空的DGL图
    g = atom_to_graph(Smile,encoder_atom)
    
    if g != False:
        return g
    else:
        print('Cannot solve SMILES:', Smile)
        return False
        
def Similes_Graph_TUD(Smile, atom_map):
    #generate graph
    # 创建一个空的DGL图
    g = atom_to_graph_TUDataset(Smile, atom_map)
    
    if g != False:
        return g
    else:
        print('Cannot solve SMILES:', Smile)
        return False
 
 
def graphs_coor(graphs, filename, output_dir='./output_coordinates'):
    # 组合输出目录和文件名
    full_output_dir = os.path.join(output_dir, filename)
    
    # 确保输出目录存在
    os.makedirs(full_output_dir, exist_ok=True)
 
    # 遍历所有图
    for idx, graph in enumerate(graphs):
        # 获取节点坐标
        coordinates = graph.pos
        
        # 将节点坐标写入到文件
        output_path = os.path.join(full_output_dir, f'graph_{idx}_coordinates.txt')
        with open(output_path, 'w') as file:
            for coordinate in coordinates:
                # 假设每个节点的坐标是一个具有两个元素的列表或数组
                file.write(f"{coordinate[0]} {coordinate[1]}\n")
 
    print(f"All coordinates have been saved in {full_output_dir}.")


def process_smiles_from_csv(dataset, encoder_atom, label_names):
    df = pd.read_csv('./data/'+dataset+'.csv')
    smiles_list = df['smiles'].tolist()
    labels = df[label_names].values
    
    output_dir = './data/'+dataset
    coordinates_dir = output_dir + '/coordinates'
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(coordinates_dir, exist_ok=True)
    
    data_list = []
    
    for i, smiles in enumerate(smiles_list):
        # 将SMILES转化为图
        try:
            graph_data = Similes_Graph(smiles, encoder_atom)
        except Exception as e:
            # 打印出错误信息和发生错误的 SMILES
            print(f"Error processing SMILES: {smiles}")
            print(f"Exception: {e}")
        
        if graph_data:
            # 保存坐标到 i.txt
            coord_file = os.path.join(coordinates_dir, f'{i}.txt')
            with open(coord_file, 'w') as f:
                for coord in graph_data.pos:
                    f.write(f"{coord[0]} {coord[1]} {coord[2]}\n")
            
            # 提取第 i 行的标签并转换为张量
            y = torch.tensor(labels[i], dtype=torch.float32)
            
            # 准备要保存的Data对象，包含图结构和标签
            data = Data(x=graph_data.x, edge_index=graph_data.edge_index, y=y, index=torch.tensor([i]))
            
            # 将Data对象添加到列表中
            data_list.append(data)
    
    # 将整个data_list保存为一个文件
    torch.save(data_list, os.path.join(output_dir, 'Graph_'+dataset+'.pt'))
    
def process_smiles_from_csv_TUD(dataset, atom_map, label_names):
    df = pd.read_csv('./data/'+dataset+'.csv')
    smiles_list = df['smiles'].tolist()
    labels = df[label_names].values
    
    output_dir = './data/'+dataset
    coordinates_dir = output_dir + '/coordinates'
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(coordinates_dir, exist_ok=True)
    
    data_list = []
    
    for i, smiles in enumerate(smiles_list):
        # 将SMILES转化为图
        try:
            graph_data = Similes_Graph_TUD(smiles, atom_map)
        except Exception as e:
            # 打印出错误信息和发生错误的 SMILES
            print(f"Error processing SMILES: {smiles}")
            print(f"Exception: {e}")
        
        if graph_data:
            # 保存坐标到 i.txt
            coord_file = os.path.join(coordinates_dir, f'{i}.txt')
            with open(coord_file, 'w') as f:
                for coord in graph_data.pos:
                    f.write(f"{coord[0]} {coord[1]} {coord[2]}\n")
            
            # 提取第 i 行的标签并转换为张量
            y = torch.tensor(labels[i], dtype=torch.float32)
            
            # 准备要保存的Data对象，包含图结构和标签
            data = Data(x=graph_data.x, edge_index=graph_data.edge_index, y=y, index=torch.tensor([i]))
            
            # 将Data对象添加到列表中
            data_list.append(data)
    
    # 将整个data_list保存为一个文件
    torch.save(data_list, os.path.join(output_dir, 'Graph_'+dataset+'.pt'))

    