import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv
from torch.autograd import Variable
import sys
sys.path.append("models/")


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_mlp_layers):
        super(MLP, self).__init__()
        
        # ��ʼ�����б�
        layers = []
        
        # �����
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())  # ʹ�� ReLU ��Ϊ�����
        
        # ���ز�
        for _ in range(num_mlp_layers - 2):  # �м�����ز�
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # �����
        layers.append(nn.Linear(hidden_dim, output_dim))  # ���������ά��
        
        # ʹ�� nn.Sequential �����в����
        self.fc = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.fc(x)

class GraphCNN(nn.Module):
    def __init__(self, RT_matrices, RT_edge_indexes, num_pooling_layers, input_dim, output_dim, final_dropout, device):
        '''
            RT_matrices: rhomboid tiling clustering matrix
            RT_edge_indexes: updated edge indexes after pooling
            num_pooling_layers: number of layers in the rhomboid tiling pooling process
            input_dim: dimensionality of input features
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            device: which device to use
        '''

        super(GraphCNN, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        
        ###Rhomboid tiling information
        #num_pooling_layers = 2
        print('# pooling layers is', num_pooling_layers)
        self.num_pooling_layers = num_pooling_layers
        self.RT_matrices = RT_matrices
        self.RT_edge_indexes = RT_edge_indexes
        self.gcn = GCNConv(input_dim, input_dim)
        self.RT_gcns = nn.ModuleList([GCNConv(input_dim, input_dim) for _ in range(num_pooling_layers)])
        self.mlps = nn.ModuleList([MLP(input_dim,input_dim,input_dim,2) for _ in range(num_pooling_layers)])
        self.RT_gins = nn.ModuleList([GINConv(self.mlps[i]) for i in range(num_pooling_layers)])
        #self.RT_gcns2 = nn.ModuleList([GCNConv(input_dim, input_dim) for _ in range(num_pooling_layers)])
        
        self.relu = nn.ReLU()
        
        self.w = nn.Parameter(torch.randn(input_dim, 1))

        ###List of MLPs
        self.mlps = torch.nn.ModuleList()
        self.sigmoid = nn.Sigmoid()

        self.linear1 = nn.Linear(input_dim, output_dim)

    def RT_pool(self, batch_graph):
        graph_representations = []
    
        for i, graph in enumerate(batch_graph):
            #### original GCN layers
            if graph.edge_index.numel() == 0:
                graph.edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(self.device)
            try:
                x = self.gcn(graph.x.to(self.device), graph.edge_index.to(self.device))
            except Exception as e:
                print("Error during GCN forward pass:", e)
                print("graph.x shape:", graph.x.shape)
                print("graph.edge_index shape:", graph.edge_index.shape)
                print("SMILE index:",graph.index.item())
                raise
                
            #### Rhomboid tiling pooling layers
            for j in range(self.num_pooling_layers):
                if isinstance(graph.index, torch.Tensor):
                    idx = graph.index.item()
                else:
                    idx = graph.index
                #RT_gcn = self.RT_gcns[j]
                #RT_gcn2 = self.RT_gcns2[j]
                RT_gin = self.RT_gins[j]
                RT_mat = torch.tensor(self.RT_matrices[idx][j], dtype=torch.float).to(self.device)
                RT_edge_indexes = self.RT_edge_indexes[idx][j+1].clone().detach().to(torch.int64).to(self.device)
                
                if RT_mat.numel() > 0:
                    try:
                        pooled_x = torch.matmul(RT_mat, x)
                        x = RT_gin(pooled_x, RT_edge_indexes)
                    except Exception as e:
                        print("Error during GCN forward pass:", e)
                        print("graph.x shape:", graph.x.shape)
                        print("x shape", x.shape)
                        print("SMILE index:", idx)
                        raise
                else:
                    pooled_x = x
                    x = RT_gin(pooled_x, torch.tensor([[0], [0]], dtype=torch.long).to(self.device))

                    
                x = self.relu(x)
                #x = RT_gcn2(x, RT_edge_indexes)
                #x = self.relu(x)
    
            # ���һ������ mean pooling
            #mean_pooled_x = torch.mean(x, dim=0)  # ��ÿ��ͼ�еĽڵ��������о�ֵ�ػ�
            mean_pooled_x = torch.matmul(x.T, torch.matmul(x,self.w)).view(-1)
            
            graph_representations.append(mean_pooled_x)  # �洢ÿ��ͼ�ı�ʾ
    
        # ���� batch ������ͼ�ı�ʾ
        return torch.stack(graph_representations)  # ������ͼ�ı�ʾ�ѵ���һ������                
                

    def forward(self, batch_graph):
        RT_represnetations = self.RT_pool(batch_graph)
 
        score = F.dropout(self.linear1(RT_represnetations), self.final_dropout, training=self.training)

        #out = self.sigmoid(score)
        return score
