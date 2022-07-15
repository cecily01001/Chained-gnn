import dgl
import click
import torch
from pandas.core.frame import DataFrame
from torch import nn as nn
from sklearn.manifold import TSNE
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import *
import numpy as np
from torch.nn import functional as F
from utils.utilsformalware import PREFIX_TO_Malware_ID, ID_TO_Malware
from utils.utilsforapps import PREFIX_TO_APP_ID,ID_TO_APP
from utils.utilsforentropy import PREFIX_TO_ENTROPY_ID,ID_TO_ENTROPY
from torch.utils.data import DataLoader
from dgl.nn import SGConv, GATConv, TAGConv, GlobalAttentionPooling, AvgPooling, GraphConv,MaxPooling,SumPooling
from dgl.data.utils import load_graphs
from torch.utils.data.dataset import ConcatDataset

def normalise_cm(cm):
    with errstate(all='ignore'):
        normalised_cm = cm / cm.sum(axis=1, keepdims=True)
        normalised_cm = nan_to_num(normalised_cm)
        return normalised_cm


def plot_confusion_matrix(cm, labels, num):
    normalised_cm = normalise_cm(cm)
    fig, ax = plt.subplots(figsize=(num, num))
    sns.heatmap(
        data=normalised_cm, cmap='YlGnBu',
        xticklabels=labels, yticklabels=labels,
        annot=True, ax=ax, fmt='.2f'
    )
    ax.set_xlabel('Predict labels')
    ax.set_ylabel('True labels')
    return fig


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


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = SGConv(in_dim, hidden_dim)
        self.conv2 = SGConv(hidden_dim, 256)
        # self.conv3 = SGConv(int(hidden_dim / 2), int(hidden_dim / 2))
        # self.conv1 = SGConv(in_dim, int(hidden_dim / 2), k=2)
        # self.conv3 = SGConv(256, 126)
        # self.conv1 = TAGConv(in_dim, hidden_dim)
        # self.conv2 = TAGConv(hidden_dim, int(hidden_dim/2))
        # self.conv1 = GraphConv(in_dim, hidden_dim)
        # self.conv2 = GraphConv(hidden_dim, 256)
        # self.conv1 = GATConv(in_dim, hidden_dim, num_heads=1)
        # self.conv2 = GATConv(hidden_dim, int(hidden_dim / 2), num_heads=1)
        # gate_nn = nn.Linear(516, 1)
        self.pooling = AvgPooling()
        # self.l1 = nn.Linear(int(hidden_dim / 2), int(hidden_dim / 4))
        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, n_classes)

    def forward(self, g):
        # 使用节点的入度作为初始特征
        h = g.ndata['h'].float()
        # print(h)
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        # h = F.relu(self.conv3(g, h))
        # h= torch.flatten(h,1)
        # h = F.relu(self.conv3(g, h))
        # h=self.fc1(h)
        # h=self.fc2(h)
        g.ndata['h'] = h  ## 节点特征经过两层卷积的输出
        # hg = dgl.mean_nodes(g, 'h')  # 图的特征是所有节点特征的均值
        hg = self.pooling(g, h)
        hg = F.relu(self.l1(hg))
        y = self.l2(hg)
        # y=self.pooling
        return y

use_gpu = torch.cuda.is_available()

print('use gpu:', use_gpu)
device = torch.device("cuda:0")

def train_graphs(train, test,kind):
    train_graphs, train_labels = load_graphs(test)
    print("load train graphs", test)
    test_graphs, test_labels = load_graphs(train)
    print("load valid graphs", train)
    # train_graphs3, train_labels3 = load_graphs(train3)
    # print("load test graphs", train3)
    # train_graphs4, train_labels4 = load_graphs(train4)
    # print("load test graphs", train4)
    # train_graphs5, train_labels5 = load_graphs(train5)
    # print("load test graphs", train5)
    length = 1500
    if kind == 'app':
        label_dic = ID_TO_APP
    elif kind == 'malware':
        label_dic = ID_TO_Malware
    else:
        label_dic = ID_TO_ENTROPY

    trainset = graphdataset(train_graphs, train_labels['labels'].numpy().tolist(), len(label_dic))
    print("trainset done")
    testset = graphdataset(test_graphs, test_labels['labels'].numpy().tolist(), len(label_dic))
    print("validset done")
    # concat_dataset = ConcatDataset([trainset2, trainset3,trainset4])
    data_loader = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=collate)
    model = Classifier(length, 512, trainset.num_classes)
    loss_func = nn.CrossEntropyLoss()

    if (use_gpu):
        model = model.to(device)
        loss_func = loss_func.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    min_loss = 1000
    avoid_over = 0
    epoch_file = open(kind + 'result/epoch_process.csv', 'w', encoding='utf-8', newline="")
    losscsv = csv.writer(epoch_file)
    losscsv.writerow(["num", "epoch"])
    # time_file = open(kind + 'result/train.csv', 'w', encoding='utf-8', newline="")
    # time_writer = csv.writer(time_file)
    # time_writer.writerow(["epoch", "time"])
    # last_loss = 10
    for epoch in range(1000):
        model.train()
        epoch_loss = 0
        for iter, (bg, label) in enumerate(data_loader):
            if (use_gpu):
                bg, label = bg.to(device), label.to(device)
            prediction = model(bg)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        if epoch_loss < 5 and epoch_loss > 0.0004:
            avoid_over = 0
            model.eval()
            valid_x, valid_y = map(list, zip(*testset))
            # valid_bg = dgl.batch(valid_x)
            # if(use_gpu):
            valid_bg = dgl.batch(valid_x).to(device)
            valid_y = torch.tensor(valid_y).float().view(-1, 1).to(device)
            after =model(valid_bg)
            valid_probs_Y = torch.softmax(after, 1)

            # valid_sampled_Y = torch.multinomial(valid_probs_Y, 1)
            valid_argmax_Y = torch.max(valid_probs_Y, 1)[1].view(-1, 1)
            # accuracy1 = (valid_y == valid_sampled_Y.float()).sum().item() / len(valid_y) * 100
            accuracy2 = (valid_y == valid_argmax_Y.float()).sum().item() / len(valid_y) * 100
            # valid_sampled_Y_done = valid_sampled_Y.float()
            sum = [0] * len(label_dic)
            index = [0] * len(label_dic)
            pred_sum = [0] * len(label_dic)
            for i in range(len(valid_x)):
                sum[int(valid_y[i, 0])] = sum[int(valid_y[i, 0])] + 1
                pred_sum[int(valid_argmax_Y[i, 0])] = pred_sum[int(valid_argmax_Y[i, 0])] + 1
                if valid_y[i] == valid_argmax_Y.float()[i]:
                    index[int(valid_y[i, 0])] = index[int(valid_y[i, 0])] + 1

            # print('Epoch {},Accuracy of sampled predictions on the valid set: {:.4f}%'.format(epoch, accuracy1))
            print('Epoch {},Accuracy of argmax predictions on the valid set: {:.4f}%'.format(epoch, accuracy2))
            if accuracy2 > 85 and epoch_loss < min_loss :
                min_loss = epoch_loss
                data1 = DataFrame(after)
                data1.to_csv('/home/user1/PangBo/CGNN/TSNE/after_app.csv', index=False, header=False)
                data2 = DataFrame(valid_y)
                data2.to_csv('/home/user1/PangBo/CGNN/TSNE/after_app_label.csv', index=False, header=False)
                torch.save(model, kind + 'result/model.pkl')
                print("model saved")
                cm = zeros((trainset.num_classes, trainset.num_classes), dtype=float)
                argmax_Y_done = valid_argmax_Y.float()
                for i in range(len(valid_y)):
                    cm[int(valid_y[i, 0]), int(argmax_Y_done[i, 0])] += 1
                sum = [0] * len(label_dic)
                index = [0] * len(label_dic)
                pred_sum = [0] * len(label_dic)
                for i in range(len(valid_y)):
                    sum[int(valid_y[i, 0])] = sum[int(valid_y[i, 0])] + 1
                    pred_sum[int(argmax_Y_done[i, 0])] = pred_sum[int(argmax_Y_done[i, 0])] + 1
                    if valid_y[i] == argmax_Y_done.float()[i]:
                        index[int(valid_y[i, 0])] = index[int(valid_y[i, 0])] + 1
                j = 0
                print("Accuracy of argmax predictions")
                f = open(kind + 'result/valid_precision_and_recall.csv', 'w', encoding='utf-8', newline="")
                valid_writer = csv.writer(f)
                valid_writer.writerow(["app", "recall", "precision", "f1 score"])
                for i in range(len(sum)):
                    if sum[j] is not 0 and pred_sum[j] is not 0:
                        recall = index[j] / sum[j]
                        precision = index[j] / pred_sum[j]
                        print(str(i) + ' kind recall is: ' + str(recall))
                        print(str(i) + ' kind precision is: ' + str(precision))
                        if ((precision+recall)!=0):
                            f1score = 2 * recall * precision / (precision + recall)
                        else:f1score=0
                    else:
                        recall = 0
                        precision = 0
                        f1score=0
                    app_name = label_dic.get(i)
                    valid_writer.writerow([app_name, recall, precision,f1score])
                    j = j + 1
                app_labels = []
                for i in sorted(list(label_dic.keys())):
                    app_labels.append(label_dic[i])
                fig = plot_confusion_matrix(cm, app_labels, len(label_dic))
                fig.savefig(kind + 'result/valid_heatmap.png')

        elif epoch_loss <= 0.0004:
            avoid_over = avoid_over + 1
            if avoid_over > 10:
                break
    model = torch.load(kind + 'result/model.pkl')
    test_X, test_Y = map(list, zip(*testset))
    test_bg = dgl.batch(test_X)

    if(use_gpu):
        test_bg = dgl.batch(test_X).to(device)
    test_Y = torch.tensor(test_Y).float().view(-1, 1).to(device)


    tsne_2D = TSNE(n_components=3, init='pca', random_state=0)  # 调用TSNE
    before = []
    for i in test_X:
        before.append(np.array(i.ndata['h'].numpy().tolist()).mean(axis=0))
    print(before)
    before_3D = tsne_2D.fit_transform(before)
    data_before = DataFrame(before_3D)
    data_before.to_csv('/home/user1/PangBo/CGNN/TSNE/before_en_3D.csv', index=False, header=False)
    after=model(test_bg)
    after.to(device)
    probs_Y = torch.softmax(after, 1)

    # probs_Y = torch.softmax(model(test_bg), 1)
    # sampled_Y = torch.multinomial(probs_Y, 1)
    argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
    # app_test_accuracy1 = (test_Y == sampled_Y.float()).sum().item() / len(test_Y)
    app_test_accuracy2 = (test_Y == argmax_Y.float()).sum().item() / len(test_Y)
    # print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(
    #     (test_Y == sampled_Y.float()).sum().item() / len(test_Y) * 100))
    print('Accuracy of argmax predictions on the test set: {:4f}%'.format(
        (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))
    accuracyfile = open(kind + 'result/accuracyfile.csv', 'w', encoding='utf-8', newline="")
    accuracyfile_writer = csv.writer(accuracyfile)
    accuracyfile_writer.writerow(["num", "accuracy"])
    # accuracyfile_writer.writerow(["sampled", app_test_accuracy1])
    accuracyfile_writer.writerow(["argmax", app_test_accuracy2])
    cm = zeros((trainset.num_classes, trainset.num_classes), dtype=float)

    # if app_test_accuracy1 >= app_test_accuracy2:
    #     sampled_Y_done = sampled_Y.float()
    #     for i in range(len(test_Y)):
    #         cm[int(test_Y[i, 0]), int(sampled_Y_done[i, 0])] += 1
    #     sum = [0] * len(label_dic)
    #     index = [0] * len(label_dic)
    #     pred_sum = [0] * len(label_dic)
    #     for i in range(len(test_Y)):
    #         sum[int(test_Y[i, 0])] = sum[int(test_Y[i, 0])] + 1
    #         pred_sum[int(sampled_Y_done[i, 0])] = pred_sum[int(sampled_Y_done[i, 0])] + 1
    #         if test_Y[i] == sampled_Y.float()[i]:
    #             index[int(test_Y[i, 0])] = index[int(test_Y[i, 0])] + 1
    #     j = 0
    #     print("Accuracy of sampled predictions")
    #     f = open(kind + 'result/precision_and_recall.csv', 'w', encoding='utf-8', newline="")
    #     csv_writer = csv.writer(f)
    #     csv_writer.writerow(["app", "recall", "precision", "f1 score"])
    #     for i in range(len(sum)):
    #         if sum[j] is not 0 and pred_sum[j] is not 0:
    #             recall = index[j] / sum[j]
    #             precision = index[j] / pred_sum[j]
    #             print(str(i) + ' kind recall is: ' + str(recall))
    #             print(str(i) + ' kind precision is: ' + str(precision))
    #             f1score = 2 * recall * precision / (precision + recall)
    #         else:
    #             recall = 0
    #             precision = 0
    #             f1score = 0
    #         app_name = label_dic.get(i)
    #         csv_writer.writerow([app_name, recall, precision,f1score])
    #         j = j + 1
    # else:
    argmax_Y_done = argmax_Y.float()
    for i in range(len(test_Y)):
        cm[int(test_Y[i, 0]), int(argmax_Y_done[i, 0])] += 1
    sum = [0] * len(label_dic)
    index = [0] * len(label_dic)
    pred_sum = [0] * len(label_dic)
    for i in range(len(test_Y)):
        sum[int(test_Y[i, 0])] = sum[int(test_Y[i, 0])] + 1
        pred_sum[int(argmax_Y_done[i, 0])] = pred_sum[int(argmax_Y_done[i, 0])] + 1
        if test_Y[i] == argmax_Y_done.float()[i]:
            index[int(test_Y[i, 0])] = index[int(test_Y[i, 0])] + 1
    j = 0
    print("Accuracy of argmax predictions")
    f = open(kind + 'result/precision_and_recall.csv', 'w', encoding='utf-8', newline="")
    csv_writer = csv.writer(f)
    csv_writer.writerow(["app", "recall", "precision","f1 score"])

    for i in range(len(sum)):
        if sum[j] is not 0 and pred_sum[j] is not 0:
            recall = index[j] / sum[j]
            precision = index[j] / pred_sum[j]
            print(str(i) + ' kind recall is: ' + str(recall))
            print(str(i) + ' kind precision is: ' + str(precision))
            if (precision+recall)!=0:
                f1score=2*recall*precision/(precision+recall)
            else:
                f1score=0
        else:
            recall = 0
            precision = 0
            f1score=0
        app_name = label_dic.get(i)
        csv_writer.writerow([app_name, recall, precision,f1score])
        j = j + 1
    app_labels = []
    for i in sorted(list(label_dic.keys())):
        app_labels.append(label_dic[i])
    fig = plot_confusion_matrix(cm, app_labels, len(label_dic))
    fig.savefig(kind + 'result/heatmap.png')

    result_2D = tsne_2D.fit_transform(after.cpu().detach().numpy())
    data1 = DataFrame(result_2D)
    data1.to_csv('/home/user1/PangBo/CGNN/TSNE/after_en.csv', index=False, header=False)
    data2=[]
    for i in test_Y:
        # print(str(int(i)))
        data2.append(label_dic.get(int(i)))
    data2 = DataFrame(data2)
    data2.to_csv('/home/user1/PangBo/CGNN/TSNE/after_en_label.csv', index=False, header=False)

@click.command()
@click.option('-k', '--kind', help='object to be classificated', required=True)
def main(kind):
    if kind == 'app':
        train_graphs("Graphs/app5fold/2_trainset.bin", "Graphs/app5fold/2_testset.bin","app")
    elif kind == 'malware':
        train_graphs("Graphs/malware/trainset.bin", "Graphs/malware/validset.bin", "Graphs/malware/testset.bin","malware")
    else:
        train_graphs("Graphs/entropy/trainset.bin", "Graphs/entropy/validset.bin", "Graphs/entropy/testset.bin",
                     "entropy")

if __name__ == '__main__':
    main()
