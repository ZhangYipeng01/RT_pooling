import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch_geometric.datasets import TUDataset

from tqdm import tqdm

from util import load_data, separate_data, separate_TUDataset
from models.RTpooling import GraphCNN

from config import label_names

from torch_geometric.data import Data
class MyData(Data):
    def __init__(self, index=None, **kwargs):
        super().__init__(**kwargs)
        self.index = index

criterion = nn.CrossEntropyLoss()

def train(args, model, device, train_graphs, optimizer, epoch):
    model.train()

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch', disable=True)

    loss_accum = 0
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        output = model(batch_graph)

        labels = torch.LongTensor([graph.y.item() for graph in batch_graph]).to(device)

        #compute loss
        loss = criterion(output, labels)

        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()         
            optimizer.step()
        

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        #report
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum/total_iters
    print("loss training: %f" % (average_loss))
    
    return average_loss

def pass_data_iteratively(model, graphs, minibatch_size = 64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)

def test(args, model, device, train_graphs, test_graphs, epoch):
    model.eval()

    output = pass_data_iteratively(model, train_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.y.item() for graph in train_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.y.item() for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    print("accuracy train: %f test: %f" % (acc_train, acc_test))

    return acc_train, acc_test

def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=9,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_pooling_layers', type=int, default=2,
                        help='number of pooling layers should smaller than max_order (default: 2)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--filename', type = str, default = "",
                                        help='output file')
    args = parser.parse_args()

    #set up seeds and gpu device
    random_seed = 0
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    print('Current Dataset is ', args.dataset)
    graphs = torch.load('./data/'+args.dataset+'/Graph_'+args.dataset+'.pt')

    num_classes = 2
    print('Num classes:',num_classes)

    train_graphs, test_graphs = separate_TUDataset(graphs, args.seed, args.fold_idx)

    
    data_RT_matrices = torch.load('./data/'+args.dataset+'/data_RT_matrices.pt')
    data_RT_edge_indexes = torch.load('./data/'+args.dataset+'/Gen_edge_indexes.pt')
    #data_RT_edge_indexes = torch.load('./data/'+args.dataset+'/Delaunay_edge_indexes.pt')
    #for idx in range(len(data_RT_edge_indexes)):
        #print('Edge index:',data_RT_edge_indexes[idx])
    
    model = GraphCNN(data_RT_matrices, data_RT_edge_indexes, args.num_pooling_layers, train_graphs[0].x.shape[1],  num_classes, args.final_dropout, device).to(device)

    weight_decay = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

    max_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print("Current epoch is:", epoch)
        scheduler.step()

        avg_loss = train(args, model, device, train_graphs, optimizer, epoch)
        acc_train, acc_test = test(args, model, device, train_graphs, test_graphs, epoch)

        max_acc = max(max_acc, acc_test)

        if not args.filename == "":
            with open(args.filename, 'a+') as f:
                f.write("%f %f %f" % (avg_loss, acc_train, acc_test))
                f.write("\n")
        print("The best test accuracy is: ", max_acc)

    with open(str(args.dataset)+'_acc_results.txt', 'a+') as f:
        f.write(str(max_acc) + '\n')
    

if __name__ == '__main__':
    main()

