import numpy as np
import os
from config import rhomboidtiling_path, label_names
from create_data import process_smiles_from_csv_TUD
import argparse

parser = argparse.ArgumentParser(
        description='RTpooling for whole-graph classification')
parser.add_argument('--dataset', type=str, default="MUTAG",
                    help='name of dataset (default: MUTAG)')
args = parser.parse_args()

dataset = args.dataset
if dataset == 'PTC_MR':
    atom_map = { 0: "In", 1: "P", 2: "O", 3: "N", 4: "Na", 5: "C", 6: "Cl", 7: "S",
        8: "Br", 9: "F", 10: "K", 11: "Cu", 12: "Zn", 13: "I", 14: "Ba",
        15: "Sn", 16: "Pb", 17: "Ca"}

if dataset == 'PTC_MM':
    atom_map = {0: "In", 1: "P", 2: "O", 3: "N", 4: "Na", 5: "C", 6: "Cl", 7: "S",
        8: "Br", 9: "F", 10: "As", 11: "K", 12: "B", 13: "Cu", 14: "Zn",
        15: "I", 16: "Ba", 17: "Sn", 18: "Pb", 19: "Ca"}

if dataset == 'PTC_FR':
    atom_map = {0: "In", 1: "P", 2: "O", 3: "N", 4: "Na", 5: "C", 6: "Cl", 7: "S",
        8: "Br", 9: "F", 10: "As", 11: "K", 12: "Cu", 13: "Zn", 14: "I", 15: "Sn", 16: "Pb", 17: "Te", 18: "Ca"}

if dataset == 'PTC_FM':
    atom_map = {0: "In", 1: "P", 2: "C", 3: "O", 4: "N", 5: "Cl", 6: "S", 7: "Br", 8: "Na", 9: "F", 10: "As", 11: "K",
                12: "Cu", 13: "I", 14: "Ba", 15: "Sn", 16: "Pb", 17: "Ca"}

if dataset == 'MUTAG':
    atom_map = {0: "C", 1: "N", 2: "O", 3: "F", 4: "I", 5: "Cl", 6: "Br"}

current_dir = os.path.dirname(os.path.abspath(__file__))
pre = current_dir + '/data/' + dataset + '/'

if not os.path.exists(pre):
    os.makedirs(pre)

if not os.path.exists(pre+'coordinates'):
    os.makedirs(pre+'coordinates')
process_smiles_from_csv_TUD(dataset, atom_map, label_names)

if not os.path.exists(pre+'filtrations'):
    os.makedirs(pre+'filtrations')
     
coords_path = pre + 'coordinates/'
filtrations_path = pre + 'filtrations/'
item_list = os.listdir(coords_path)
for item in item_list:
    item_path = os.path.join(coords_path, item)
    if os.path.isfile(item_path) and item.endswith(".txt"):
        file_name = os.path.splitext(item)[0]
        os.system(
            "cd " + rhomboidtiling_path + " && ./main "+ item_path +" "+ filtrations_path + file_name +"_fslices.txt 3 4 fslices")
        os.system(
            "cd " + rhomboidtiling_path + " && ./main " + item_path + " " + filtrations_path + file_name + "_rhomboids.txt 3 4 rhomboids")
            
print('All rhomboid tiling files are succussfully generated')