import dgl
import click
import torch
from torch import nn as nn
import time
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import *
from torch.nn import functional as F
from utils.utilsformalware import PREFIX_TO_Malware_ID
from utils.utilsforapps import PREFIX_TO_APP_ID
from utils.utilsforentropy import PREFIX_TO_ENTROPY_ID
from torch.utils.data import DataLoader
from dgl.nn import SGConv, GATConv, TAGConv, GlobalAttentionPooling, AvgPooling, GraphConv
from dgl.data.utils import load_graphs


def normalise_cm(cm):
    with errstate(all='ignore'):
        normalised_cm = cm / cm.sum(axis=1, keepdims=True)
        normalised_cm = nan_to_num(normalised_cm)
        return normalised_cm

class graphdataset(object):
    def __init__(self, graphs, labels, num_classes):
        super(graphdataset, self).__init__()
        self.graphs = graphs
        self.num_classes = num_classes
        self.labels = labels

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx):
        """Get the i^th sample.

        Paramters
        ---------
        idx : int
            The sample index.

        Returns
        -------
        (dgl.DGLGraph, int)
            The graph and its label.
        """
        return self.graphs[idx], self.labels[idx]


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    # print(batched_graph)
    # print(labels)
    return batched_graph, torch.tensor(labels)

def plot_confusion_matrix(cm,nodenum,length):
    # normalised_cm = normalise_cm(cm)
    fig, ax = plt.subplots(figsize=(length, nodenum))
    sns.heatmap(
        data=cm,cmap='YlGnBu', ax=ax
    )
    return fig

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = SGConv(in_dim, hidden_dim)
        self.conv2 = SGConv(hidden_dim, 256)
        # self.conv1 = SGConv(in_dim, int(hidden_dim / 2), k=2)
        # self.conv3 = SGConv(256, 126)
        # self.conv1 = TAGConv(in_dim, hidden_dim)
        # self.conv2 = TAGConv(hidden_dim, int(hidden_dim/2))
        # self.conv1 = GraphConv(in_dim, hidden_dim)
        # self.conv2 = GraphConv(hidden_dim, 256)
        # self.gat = GATConv(hidden_dim, 256, num_heads=1)
        # gate_nn = nn.Linear(516, 1)
        self.pooling = AvgPooling()
        # self.l1 = nn.Linear(int(hidden_dim / 2), int(hidden_dim / 4))
        self.l2 = nn.Linear(256, n_classes)

    def forward(self, g):
        # 使用节点的入度作为初始特征
        h = g.ndata['h'].float()
        print('原')
        print(h)
        fig = plot_confusion_matrix(h,g.num_nodes(),655)
        fig.savefig('result/1.png')
        h = F.relu(self.conv1(g, h))
        fig = plot_confusion_matrix(h.detach().numpy(), g.num_nodes(), 256)
        fig.savefig('result/2.png')
        print('第一层')
        print(h[0][425:])
        # print(len(h[0]))
        h = F.relu(self.conv2(g, h))
        fig = plot_confusion_matrix(h.detach().numpy(), g.num_nodes(),256)
        fig.savefig('result/3.png')
        print('第二层')
        print(h)
        g.ndata['h'] = h  ## 节点特征经过两层卷积的输出
        hg = self.pooling(g, h)
        fig = plot_confusion_matrix(hg.detach().numpy(), 1, 256)
        fig.savefig('result/4.png')
        # hg = self.l1(hg)
        y = self.l2(hg)
        # y=self.pooling
        return y



use_gpu = torch.cuda.is_available()
print('use gpu:', use_gpu)
device = torch.device("cuda:0")


def train_graphs(train):
    train_graphs, train_labels = load_graphs(train)
    print("load train graphs", train)
    trainset = graphdataset(train_graphs, train_labels['labels'].numpy().tolist(), 41)
    print("trainset done")
    data_loader = DataLoader(trainset, batch_size=1, shuffle=True, collate_fn=collate)
    model = Classifier(1500, 516, trainset.num_classes)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # model = torch.load('app' + 'result/model.pkl')
    # for epoch in range(400):
    #     model.train()
    #     epoch_loss = 0
    for epoch in range(140):
        model.train()
        epoch_loss = 0
        for iter, (bg, label) in enumerate(data_loader):
            if bg.num_nodes() == 4:
                prediction = model(bg)
                fig = plot_confusion_matrix(prediction.detach().numpy(), 1, 41)
                fig.savefig('result/5.png')
                loss = loss_func(prediction, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()
                soft=torch.softmax(prediction, 1)
                print(soft)
                fig = plot_confusion_matrix(soft.detach().numpy(), 1, 41)
                fig.savefig('result/6.png')
        epoch_loss /= (iter + 1)

def main():
    train_graphs("Graphs/app/testset.bin")


if __name__ == '__main__':
    main()
