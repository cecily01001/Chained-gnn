import os
import dgl
import dgl.nn as dglnn
import numpy
import torch
from pathlib import Path
from torch import nn as nn
import torch.nn.functional as F
from dgl.data.utils import save_graphs,load_graphs,save_info,load_info
import pandas as pd
# 线性
class LinearNet(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 =  torch.nn.Linear(in_feats, hid_feats)
        self.conv2 =  torch.nn.Linear(hid_feats, out_feats)

    def forward(self, graph, inputs):
        # inputs are features of nodes
        print('神经网络输入：')
        print(inputs.float())
        # h = self.conv1(graph, inputs.float())
        h = self.conv1(inputs.float())
        print('第一层神经网络输出：')
        print(h)
        for i in range(len(h)):
            for j in range(len(h[i])):
                if numpy.isnan(h[i][j]):
                    print('null')
                    h[i][j]=0
        h = F.relu(h)
        h = self.conv2(h)
        # h = self.conv2(graph, h)
        print('第二层神经网络输出:')
        print(h[:5])
        return h
# 神经网络模型
class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        # 实例化SAGEConve，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregator_type是聚合函数的类型
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        # 输入是节点的特征
        print('神经网络输入：')
        print(inputs.float())
        # for i in range(len(inputs.float())):
        #     for j in range(len(inputs.float()[i])):
        #         if inputs.float()[i][j].isnan():
        #             print('nan')
        #             inputs.float()[i][j]=0
        # h = self.conv1(graph, inputs.float())
        h = self.conv1(graph,inputs.float())
        print('第一层神经网络输出：')
        print(h)
        # for i in range(len(h)):
        #     for j in range(len(h[i])):
        #         if h[i][j].isnan():
        #             print('null')
        #             h[i][j]=0
        h = F.relu(h)
        h = self.conv2(graph,h)
        # h = self.conv2(graph, h)
        print('第二层神经网络输出:')
        print(h)
        return h

def train_topo(file_path):
    if not os.path.exists("Graphs_topo/topo.bin"):
        print('Create graph')
        file_path = Path(file_path)
        f = open(file_path, 'r', encoding='utf-8', newline="")
        file_contents_lines = f.readlines()
        dgllistSrc = []
        dgllistDst = []
        features = []
        node_labels = []
        for line in file_contents_lines:
            packets = line.split('] ')
        #从文件中提取顶点特征
            feature_temp=packets[0]
            # print(feature_temp)
            node_label=packets[1]
            feature_temp=feature_temp[1:len(feature_temp)]
            # print(feature_temp)
            feature = feature_temp.split(' ')
            # print(feature)
            # 从字符串转成float，放入特征矩阵
            feature_temp=list(map(float, feature))
            for i in range(len(feature_temp)):
                if numpy.isnan(feature_temp[i]):
                    print('nan')
                    feature_temp[i]=0
            features.append(feature_temp)

        #从文件中提取顶点的值value 放入顶点值矩阵
            node_labels.append([float(node_label)])
            # print(node_labels)
        # 绘制chained graph
        # 第一次读取流量数据，建图并存储

        for i in range(len(node_labels) - 1):
            dgllistSrc.append(i)
            dgllistDst.append(i + 1)
        for i in range(len(node_labels) - 1):
            dgllistSrc.append(i + 1)
            dgllistDst.append(i)
        # 建图
        node_graph = dgl.graph((dgllistSrc, dgllistDst))
        # 输入顶点特征和顶点值
        node_graph.ndata['feature'] = torch.tensor(features)
        node_graph.ndata['label'] = torch.tensor(node_labels)
        save_graphs("Graphs_topo/topo.bin", [node_graph])
        save_info("Graphs_topo/topo.pkl", {'nodeLabel': torch.tensor(node_labels),'nodeFeature':torch.tensor(features)})
    else:
        # 建图之后，直接读取图，省去反复读取数据时间
        node_graphs=load_graphs("Graphs_topo/topo.bin")
        node_graph = node_graphs[0][0]
        node_graph.ndata['feature']=load_info("Graphs_topo/topo.pkl")['nodeFeature']
        node_graph.ndata['label'] = load_info("Graphs_topo/topo.pkl")['nodeLabel']
        print('Load graph')
    # 使用90%的数据作为训练集
    node_graph.ndata['train_mask'] = torch.zeros(len(node_graph.ndata['feature']), dtype=torch.bool).bernoulli(0.9)
    node_graph.ndata['test_mask'] = ~node_graph.ndata['train_mask']
    node_features = node_graph.ndata['feature']
    node_labels = node_graph.ndata['label']
    train_mask = node_graph.ndata['train_mask']
    test_mask = node_graph.ndata['test_mask']
    # 输出原始的顶点特征
    print('未开始训练时的顶点特征：')
    print(node_features)
    for i in range(len(node_features)):
        for j in range(len(node_features[i])):
            if numpy.isnan(node_features[i][j]):
                print('nan')
                node_features[i][j]=0
    # 设置模型参数
    model=SAGE(in_feats=8, hid_feats=5, out_feats=1)
    # model = LinearNet(in_feats=8, hid_feats=5, out_feats=1)
    # opt = torch.optim.Adam(model.parameters())

    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    mse_loss = nn.MSELoss()
    for epoch in range(10):
        pred = model(node_graph, node_features)
        print('真实的node值（前6个顶点的）：')
        print(node_labels[0:5])

        # 因为出现Nan无法继续计算loss，所以设置了如果是nan则为0，仅为了能调试神经网络计算过程
        # pred = torch.where(torch.isnan(pred), torch.full_like(pred, 0), pred)
        # for i in range(len(pred)):
        #     pred[i]=torch.where(torch.isnan(pred[i]), torch.full_like(pred[i], 0), pred[i])
        #     pred[i] = torch.where(pred[i]-node_labels[i]>10, torch.full_like(pred[i], node_labels[i].item()), pred[i])

        print('第{}论训练的顶点预测值(前6个顶点）：'.format(epoch))
        print(pred[0:5])
        # 计算loss
        loss=mse_loss(pred[train_mask].to(torch.float32), node_labels[train_mask].to(torch.float32))
        print('第{}论训练的loss：{}'.format(epoch,loss))
        opt.zero_grad()
        loss.backward()
        opt.step()
        print('\n')
        print('第{}轮训练开始'.format(epoch+1))
    model.eval()
# 测试部分
    # test_pred = model(node_graph, node_features)
    # test_pred=test_pred[test_mask]
    # test_label=node_labels[test_mask].float().view(-1, 1)
    # test_pred_1 = torch.softmax(test_pred, 1)
    # # test_pred_1 = torch.multinomial(test_pred_1, 1)
    # test_pred_1=torch.max(test_pred_1, 1)[1].view(-1, 1)
    # # 模型分类准确率
    # result = (test_label == test_pred_1.float()).sum().item() / len(test_pred_1) * 100
    # # 输出模型分类准确率
    # print(result)
    # print(test_pred_1)
    # for i in range(len(test_pred_1)):
    #     if test_pred_1[i]==0:
    #         print(i)

def main():
    train_topo("tempdata/raw_D1_104_D2_360_choice_3_banks_8_types_1")

if __name__ == '__main__':
    main()
