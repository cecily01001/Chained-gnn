import csv
import pandas as pd
import click
import dgl
import jieba
import torch
from pathlib import Path
import numpy as np
from scapy.compat import raw
from scapy.all import *
from scapy.error import Scapy_Exception
from scapy.layers.inet import IP, UDP, TCP
from scapy.layers.l2 import Ether
from scapy.packet import Padding
from scipy import sparse
from utils.utilsformalware import should_omit_packet, read_pcap, PREFIX_TO_Malware_ID
from utils.utilsforapps import PREFIX_TO_APP_ID
from utils.utilsforentropy import PREFIX_TO_ENTROPY_ID, ID_TO_ENTROPY
from dgl.data.utils import save_graphs


def remove_ether_header(packet):
    if Ether in packet:
        return packet[Ether].payload

    return packet


def mask_ip(packet):
    if IP in packet:
        packet[IP].src = '0.0.0.0'
        packet[IP].dst = '0.0.0.0'

    return packet


def pad_udp(packet):
    if UDP in packet:
        # get layers after udp
        layer_after = packet[UDP].payload.copy()

        # build a padding layer
        pad = Padding()
        pad.load = '\x00' * 12

        layer_before = packet.copy()
        layer_before[UDP].remove_payload()
        packet = layer_before / pad / layer_after

        return packet

    return packet


# def packet_to_sparse_array(packet, max_length=1500):
#     arr = np.frombuffer(raw(packet), dtype=np.uint8)[0: max_length]
#     # arr = np.frombuffer(raw(packet), dtype=np.uint8)[0: max_length] / 255
#     if len(arr) < max_length:
#         pad_width = max_length - len(arr)
#         arr = np.pad(arr, pad_width=(0, pad_width), constant_values=0)
#     # print(arr)
#     arr = sparse.csr_matrix(arr)
#     return arr

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
def packet_to_sparse_array(packet, max_length=1500):
    arr = np.frombuffer(raw(packet), dtype=np.uint8)[0: max_length]
    # arr = np.frombuffer(raw(packet), dtype=np.uint8)[0: max_length] / 255
    if len(arr) < max_length:
        pad_width = max_length - len(arr)
        arr = np.pad(arr, pad_width=(0, pad_width), constant_values=0)
    arr = sparse.csr_matrix(arr)
    return arr


# def packet_to_sparse_array(packet, max_length=1500):
#     # arr = np.frombuffer(raw(packet), dtype=np.uint8)[0: max_length]
#     # arr = np.frombuffer(raw(packet), dtype=np.uint8)[0: max_length] / 255
#     arr = np.pad([], pad_width=(0, 1500), constant_values=0)
#     arr = sparse.csr_matrix(arr)
#     return arr


def transform_packet(packet, max_length):
    # if should_omit_packet(packet):
    #     return None

    packet = remove_ether_header(packet)
    packet = pad_udp(packet)
    packet = mask_ip(packet)
    arr = packet_to_sparse_array(packet, max_length)

    return arr


def transform_pcap(pcap_path, graphlist, listofkind, kind):
    feature_matrix = []
    direcs = []
    max_length = 1500
    if kind == 'app':
        label_dic = PREFIX_TO_APP_ID
    elif kind == 'malware':
        label_dic = PREFIX_TO_Malware_ID
    else:
        label_dic = PREFIX_TO_ENTROPY_ID
        # max_length = 750
    print(pcap_path)
    node_num = 0
    # p1 = int(pcap_path.name.split('.')[2].split('_')[2])
    for i, packet in enumerate(read_pcap(pcap_path)):
        arr = transform_packet(packet, max_length)
        if arr is not None:
            node_num = node_num + 1
            feature_matrix.append(arr.todense().tolist()[0])
            # direc = -1
            # if TCP in packet:
            #     sport = packet[TCP].sport
            #     if sport == p1:
            #         direc = 0
            #     else:
            #         direc = 1
            # direcs.append(direc)

    if node_num > 1:
        label = label_dic.get(pcap_path.name.split('.')[0])
        g = single_graph(node_num, feature_matrix, direcs)
        if g:
            listofkind.append(label)
            graphlist.append(g)


class graphdataset(object):
    def __init__(self, graph_list, listofkind, num_classes):
        super(graphdataset, self).__init__()
        self.graphs = graph_list
        self.num_classes = num_classes
        self.labels = listofkind
        # self.feature_matrix=feature_matrix

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


def single_graph(node_num, node_feature, direcs):
    list_point = []
    list_point_2 = []
    # process_node_num=int(node_num/10)
    # node_num=node_num+process_node_num
    for i in range(node_num - 1):
        list_point.append(i)
        list_point_2.append(i + 1)
    for i in range(node_num - 1):
        list_point.append(i + 1)
        list_point_2.append(i)
    # 加上TCP的边
    # for i in range(len(direcs) - 2):
    #     if direcs[i] == direcs[i + 1] and direcs[i] != direcs[i + 2]:
    #         temp = i + 2
    #         while (direcs[i] != direcs[i + 2]):
    #             list_point.append(temp)
    #             list_point_2.append(i)
    #             i = i - 1
    g = dgl.graph((list_point, list_point_2))
    g = dgl.add_self_loop(g)
    g.ndata['h'] = torch.tensor(node_feature, dtype=torch.float)
    # g.ndata['x'] = torch.tensor(text_feature)
    return g


def make_graph(data_dir_path, kind):
    # testdata_dir_path = Path(testdata_dir_path)
    # valid_data_dir_path = Path(valid_data_dir_path)
    data_dir_path = Path(data_dir_path)
    if kind == 'app':
        label_dic = PREFIX_TO_APP_ID
    elif kind == 'malware':
        label_dic = PREFIX_TO_Malware_ID
    else:
        label_dic = PREFIX_TO_ENTROPY_ID
    num_classes = len(label_dic)

    # 训练集
    graphlist = []
    listofkind = []
    for pcap_path in sorted(data_dir_path.iterdir()):
        transform_pcap(pcap_path, graphlist, listofkind, kind)

    f1_g = []
    f1_l = []
    f2_g = []
    f2_l = []
    f3_g = []
    f3_l = []
    f4_g = []
    f4_l = []
    f5_g = []
    f5_l = []

    temp1 = []
    temp1_l = []
    temp2 = []
    temp2_l = []
    temp3 = []
    temp3_l = []
    temp4 = []
    for i in range(len(graphlist)):
        if i % 5 == 0:
            f1_g.append(graphlist[i])
            f1_l.append(listofkind[i])
        else:
            temp1.append(graphlist[i])
            temp1_l.append(listofkind[i])
    for i in range(len(temp1_l)):
        if i % 4 == 0:
            f2_g.append(temp1[i])
            f2_l.append(temp1_l[i])
        else:
            temp2.append(temp1[i])
            temp2_l.append(temp1_l[i])
    for i in range(len(temp2_l)):
        if i % 3 == 0:
            f3_g.append(temp2[i])
            f3_l.append(temp2_l[i])
        else:
            temp3.append(temp2[i])
            temp3_l.append(temp2_l[i])
    for i in range(len(temp3_l)):
        if i % 2 == 0:
            f4_g.append(temp3[i])
            f4_l.append(temp3_l[i])
        else:
            f5_g.append(temp3[i])
            f5_l.append(temp3_l[i])

    test1 = graphdataset(f1_g, f1_l, num_classes)
    # train2 = graphdataset(f2_g, f2_l, num_classes)
    # train3 = graphdataset(f3_g, f3_l, num_classes)
    # train4 = graphdataset(f4_g, f4_l, num_classes)
    # train5 = graphdataset(f5_g, f5_l, num_classes)
    train1=graphdataset(f2_g+f3_g+f4_g+f5_g,f2_l+f3_l+f4_l+f5_l,num_classes)
    save_graphs("Graphs/" + kind + "5fold/1_trainset.bin", test1.graphs,
                {'labels': torch.tensor(test1.labels)})
    save_graphs("Graphs/" + kind + "5fold/1_testset.bin", train1.graphs,
                {'labels': torch.tensor(train1.labels)})

    test2 = graphdataset(f2_g, f2_l, num_classes)
    # train2 = graphdataset(f2_g, f2_l, num_classes)
    # train3 = graphdataset(f3_g, f3_l, num_classes)
    # train4 = graphdataset(f4_g, f4_l, num_classes)
    # train5 = graphdataset(f5_g, f5_l, num_classes)
    train2 = graphdataset(f1_g + f3_g + f4_g + f5_g, f1_l + f3_l + f4_l + f5_l, num_classes)
    save_graphs("Graphs/" + kind + "5fold/2_trainset.bin", test2.graphs,
                {'labels': torch.tensor(test2.labels)})
    save_graphs("Graphs/" + kind + "5fold/2_testset.bin", train2.graphs,
                {'labels': torch.tensor(train2.labels)})

    test3 = graphdataset(f3_g, f3_l, num_classes)
    # train2 = graphdataset(f2_g, f2_l, num_classes)
    # train3 = graphdataset(f3_g, f3_l, num_classes)
    # train4 = graphdataset(f4_g, f4_l, num_classes)
    # train5 = graphdataset(f5_g, f5_l, num_classes)
    train3 = graphdataset(f1_g + f2_g + f4_g + f5_g, f1_l + f2_l + f4_l + f5_l, num_classes)
    save_graphs("Graphs/" + kind + "5fold/3_trainset.bin", test3.graphs,
                {'labels': torch.tensor(test3.labels)})
    save_graphs("Graphs/" + kind + "5fold/3_testset.bin", train3.graphs,
                {'labels': torch.tensor(train3.labels)})

    test4 = graphdataset(f4_g, f4_l, num_classes)
    # train2 = graphdataset(f2_g, f2_l, num_classes)
    # train3 = graphdataset(f3_g, f3_l, num_classes)
    # train4 = graphdataset(f4_g, f4_l, num_classes)
    # train5 = graphdataset(f5_g, f5_l, num_classes)
    train4 = graphdataset(f1_g + f2_g + f3_g + f5_g, f1_l + f2_l + f3_l + f5_l, num_classes)
    save_graphs("Graphs/" + kind + "5fold/4_trainset.bin", test4.graphs,
                {'labels': torch.tensor(test4.labels)})
    save_graphs("Graphs/" + kind + "5fold/4_testset.bin", train4.graphs,
                {'labels': torch.tensor(train4.labels)})

    test5 = graphdataset(f5_g, f5_l, num_classes)
    # train2 = graphdataset(f2_g, f2_l, num_classes)
    # train3 = graphdataset(f3_g, f3_l, num_classes)
    # train4 = graphdataset(f4_g, f4_l, num_classes)
    # train5 = graphdataset(f5_g, f5_l, num_classes)
    train5 = graphdataset(f1_g + f2_g + f3_g + f4_g, f1_l + f2_l + f3_l + f4_l, num_classes)
    save_graphs("/home/data/PangBo/Graphs/" + kind + "5fold/5_trainset.bin", test5.graphs,
                {'labels': torch.tensor(test5.labels)})
    save_graphs("/home/data/PangBo/Graphs/" + kind + "5fold/5_testset.bin", train5.graphs,
                {'labels': torch.tensor(train5.labels)})
    # save_graphs("Graphs/" + kind + "5fold/3_trainset.bin", train3.graphs,
    #             {'labels': torch.tensor(train3.labels)})
    # save_graphs("Graphs/" + kind + "5fold/4_trainset.bin", train4.graphs,
    #             {'labels': torch.tensor(train4.labels)})
    # save_graphs("Graphs/" + kind + "5fold/5_trainset.bin", train5.graphs,
    #             {'labels': torch.tensor(train5.labels)})


@click.command()
@click.option('-k', '--kind', help='object to be classificated', required=True)
def main(kind):
    if kind == 'app':
        make_graph("Dataset/Splite_Session/app", 'app')
    elif kind == 'malware':
        make_graph("Dataset/Splite_Session/malware", 'malware')
    else:
        make_graph("Dataset/Splite_Session/encrypted", 'entropy')


if __name__ == '__main__':
    main()
