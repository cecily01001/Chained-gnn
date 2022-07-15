import csv
import os
import shutil
import dgl
import dgl.function as fn
import dgl.nn as dglnn
import torch
from pathlib import Path
import numpy as np
from click import File
from torch import nn as nn
import torch.nn.functional as F
from dgl.data.utils import save_graphs,load_graphs,save_info,load_info
from sklearn import metrics
#
class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h
# 边分类模型
class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.sage = SAGE(in_features, hidden_features, out_features)
        self.pred = MLPPredictor(out_features,2)

    def forward(self, g, x):
        h = self.sage(g, x)
        return self.pred(g, h)
#多层感知器，输出类别，就是2类（TCP/UDP）
class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(torch.cat([h_u, h_v], 1))
        return {'score': score}

    def forward(self, graph, h):
        # h是从5.1节的GNN模型中计算出的节点表示
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']

def train_topo(edge_path):
    edge_path = Path(edge_path)
    f = open(edge_path, 'r', encoding='utf-8', newline="")
    next(f)
    edges = csv.reader(f)
    # file_length=len(f.readlines())
    nodes = []
    dgllistSrc = []
    dgllistDst = []
    feature = []
    edge_label = []
    j = 0
    if not os.path.exists("Graphs_topo/topo.bin"):
        # 第一次读取流量数据，建图并存储
        print('Create graph')
        for i in edges:
            # print('process:'+str((j/file_length)*100))
            node1name = i[0].split('-')[0]
            node2name = i[0].split('-')[1]
            if node1name in nodes:
                nodeSrc = nodes.index(node1name)
            else:
                nodes.append(node1name)
                nodeSrc = nodes.index(node1name)

            if node2name in nodes:
                nodeDst = nodes.index(node2name)
            else:
                nodes.append(node2name)
                nodeDst = nodes.index(node2name)
            dgllistSrc.append(nodeSrc)
            dgllistDst.append(nodeDst)
            if (i[44] == i[45] == i[46] == i[47] == '0'):
                l = 0  # UDP
                print('0')
            else:
                l = 1  # TCP
            del i[42:46]
            del i[0]
            feature.append(list(map(float, i)))
            edge_label.append([l])
            # print(edge_label)
            j = j + 1
        # print(feature)
        edge_graph = dgl.graph((dgllistSrc, dgllistDst))
        temp = [[1]*64] * len(nodes)
        print(temp)
        print(len(temp))
        # print(edge_label)
        edge_graph.ndata['feature'] = torch.tensor(temp)
        edge_graph.edata['feature'] = torch.tensor(feature)
        edge_graph.edata['label'] = torch.tensor(edge_label)

        save_graphs("Graphs_topo/topo.bin", [edge_graph])
        save_info("Graphs_topo/topo.pkl", {'edgeLabel': torch.tensor(edge_label),'edgeFeature':torch.tensor(feature),'nodeFeature':torch.tensor(temp)})
    else:
        # 建图之后，直接读取图，省去反复读取数据时间
        edge_graphs=load_graphs("Graphs_topo/topo.bin")
        edge_graph = edge_graphs[0][0]
        print(edge_graph)
        edge_graph.ndata['feature']=load_info("Graphs_topo/topo.pkl")['nodeFeature']
        edge_graph.edata['feature'] = load_info("Graphs_topo/topo.pkl")['edgeFeature']
        edge_graph.edata['label'] = load_info("Graphs_topo/topo.pkl")['edgeLabel']
        print(edge_graph)
        print('Load graph')
    edge_graph.edata['train_mask'] = torch.zeros(len(edge_graph.edata['feature']), dtype=torch.bool).bernoulli(0.9)
    edge_graph.edata['test_mask'] = ~edge_graph.edata['train_mask']
    node_features = edge_graph.ndata['feature']
    edge_label = edge_graph.edata['label']
    train_mask = edge_graph.edata['train_mask']
    test_mask = edge_graph.edata['test_mask']
    # print(len(feature[0]))
    model = Model(64, 20, 1)
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(150):
        pred = model(edge_graph, node_features)
        # loss = ((pred[train_mask] - edge_label[train_mask]) ** 2).mean()
        # print(edge_label[train_mask])
        loss= F.cross_entropy(pred[train_mask], edge_label[train_mask].flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
        print('Epoch {}, loss {:.4f}'.format(epoch, loss.item()))
    model.eval()
    test_pred = model(edge_graph, node_features)
    test_pred=test_pred[test_mask]
    test_label=edge_label[test_mask].float().view(-1, 1)
    test_pred_1 = torch.softmax(test_pred, 1)
    # test_pred_1 = torch.multinomial(test_pred_1, 1)
    test_pred_1=torch.max(test_pred_1, 1)[1].view(-1, 1)
    # 模型分类准确率
    result = (test_label == test_pred_1.float()).sum().item() / len(test_pred_1) * 100
    # 输出模型分类准确率
    print(result)
    # print(test_pred_1)
    # for i in range(len(test_pred_1)):
    #     if test_pred_1[i]==0:
    #         print(i)

def main():
    train_topo("mawi-20m/mawi-20M.pcap_flow_record.csv")

if __name__ == '__main__':
    main()
