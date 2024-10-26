import numpy as np
import os
from config import rhomboidtiling_path
import argparse

parser = argparse.ArgumentParser(
        description='RTpooling for whole-graph classification')
parser.add_argument('--dataset', type=str, default="COX2",
                    help='name of dataset (default: COX2)')
args = parser.parse_args()
dataset = args.dataset

current_dir = os.path.dirname(os.path.abspath(__file__))
pre = current_dir + '/data/' + dataset + '/'

node_attributes = np.loadtxt(pre+dataset+'_node_attributes.txt', delimiter=',')
graph_indicator = np.loadtxt(pre+dataset+'_graph_indicator.txt', dtype=int)

if not os.path.exists(pre+'coordinates'):
    os.makedirs(pre+'coordinates')
if not os.path.exists(pre+'filtrations'):
    os.makedirs(pre+'filtrations')
    
num_graphs = np.max(graph_indicator)
for s in range(1, num_graphs + 1):
    indices = np.where(graph_indicator == s)[0]
    attributes = node_attributes[indices]
    np.savetxt(pre+'coordinates/'+str(s-1)+'.txt', attributes, fmt='%f')
    
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