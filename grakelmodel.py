import csv

import click
from pathlib import Path
from scapy.compat import raw
from scapy.all import *
from scapy.error import Scapy_Exception
from scapy.layers.inet import IP, UDP
from scapy.layers.l2 import Ether
from scapy.packet import Padding
from scipy import sparse
from utils.utilsformalware import should_omit_packet, read_pcap, PREFIX_TO_Malware_ID
from utils.utilsforapps import PREFIX_TO_APP_ID
from utils.utilsforentropy import PREFIX_TO_ENTROPY_ID, ID_TO_ENTROPY
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score
from grakel import Graph
from grakel.kernels import ShortestPath,RandomWalk,PropagationAttr,RandomWalkLabeled,GraphletSampling,ShortestPathAttr
from grakel.datasets import fetch_dataset
from sklearn.tree import DecisionTreeClassifier

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

def packet_to_sparse_array(packet, max_length=500):
    arr = np.frombuffer(raw(packet), dtype=np.uint8)[50: max_length]
    # arr = np.frombuffer(raw(packet), dtype=np.uint8)[0: max_length] / 255
    if len(arr) < max_length:
        pad_width = max_length - len(arr)
        arr = np.pad(arr, pad_width=(0, pad_width), constant_values=0)
    arr = sparse.csr_matrix(arr)
    return arr
def transform_packet(packet):
    # if should_omit_packet(packet):
    #     return None

    packet = remove_ether_header(packet)
    packet = pad_udp(packet)
    packet = mask_ip(packet)
    # wrpcap("/home/pcl/PangBo/pro/GNNTrafficClassification/Dataset/ProcessedData/" + name, packet, append=True)
    arr = packet_to_sparse_array(packet)
    return arr

def single_graph(node_num, node_feature,app_label):
    edges = {}
    node_attributes={}
    edges[0]=[1]
    edges[node_num-1]=[node_num-2]
    a=[]
    a.append(app_label)
    for i in range(node_num):
        if i!=0 and i!=node_num - 1:
            edges[i]=[i-1,i+1]
        # node_attributes[i]="".join([str(_) for _ in node_feature[i]])
        node_attributes[i] = node_feature[i]
        # node_attributes[i] = a
    # print(node_attributes)
    # if i==node_num-1:print(edges)
    # g=Graph(edges,node_labels=node_attributes)
    g = Graph(edges)
    # print(g)
    return g

def transform_pcap(pcap_path, graphlist, listofkind,dic):
    feature_matrix = []
    print(pcap_path)
    j=0
    for i, packet in enumerate(read_pcap(pcap_path)):
        arr = transform_packet(packet)
        if arr is not None:
            j = j + 1
            feature_matrix.append(arr.todense().tolist()[0])
    if j > 1:
        prefix = pcap_path.name.split('.')[0]
        app_label = dic.get(prefix)
        g=single_graph(j,feature_matrix,app_label)
        graphlist.append(g)
        listofkind.append(app_label)


def make_graph(data_dir_path, valid_data_dir_path, testdata_dir_path, kind):
    testdata_dir_path = Path(testdata_dir_path)
    valid_data_dir_path = Path(valid_data_dir_path)
    data_dir_path = Path(data_dir_path)
    if kind == 'app':
        label_dic = PREFIX_TO_APP_ID
    elif kind == 'malware':
        label_dic = PREFIX_TO_Malware_ID
    else:
        label_dic = PREFIX_TO_ENTROPY_ID
    num_classes = len(label_dic)

    # 训练集
    graphlist = list()
    listofkind = list()
    graphlist = []
    listofkind = []
    i = 0
    # for pcap_path in sorted(data_dir_path.iterdir()):
    #     if i%10==0:
    #         transform_pcap(pcap_path, graphlist, listofkind,label_dic)
    #     i+=1
    # for pcap_path in sorted(valid_data_dir_path.iterdir()):
    #     transform_pcap(pcap_path, graphlist, listofkind,label_dic)
    for pcap_path in sorted(testdata_dir_path.iterdir()):
        transform_pcap(pcap_path, graphlist , listofkind,label_dic)
    print(listofkind)
    G_train, G_test, y_train, y_test = train_test_split(graphlist, listofkind, test_size=0.3, random_state=2)

    # print(G_train)
    # Uses the shortest path kernel to generate the kernel matrices
    # gk = RandomWalk()
    gk = ShortestPath(normalize=True, with_labels=False)
    # gk=GraphletSampling()
    # gk = PropagationAttr(normalize=False)
    K_train = gk.fit_transform(G_train)
    K_test = gk.transform(G_test)

    # Uses the SVM classifier to perform classification
    # clf = SVC(kernel="precomputed")
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)
    print(y_test)
    print(y_pred)
    # Computes and prints the classification accuracy
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", str(round(acc * 100, 2)) + "%")
    f1=f1_score(y_test, y_pred,average=None)
    f = open('grakel/PropagationAttr.csv', 'w', encoding='utf-8', newline="")
    csv_writer = csv.writer(f)
    csv_writer.writerow(["f1 score"])
    for i in f1:
        csv_writer.writerow([i])

@click.command()
@click.option('-k', '--kind', help='object to be classificated', required=True)
def main(kind):
    if kind == 'app':
        make_graph("Dataset/grakel/app", "Dataset/grakel/app",
                     "Dataset/grakel/app", kind)
    #
    # elif kind == 'malware':
    #     make_graph("Dataset/Splite_Session/Malware/trainset", "Dataset/Splite_Session/Malware/validset",
    #                  "Dataset/Splite_Session/Malware/testset", kind)
    # else:
    #     make_graph("Dataset/Splite_Session/Entropy/train", "Dataset/Splite_Session/Entropy/valid",
    #                  "Dataset/Splite_Session/Entropy/test", kind)
    # ----------------------------------------------
    # MUTAG = fetch_dataset("MUTAG", verbose=False)
    # G, y = MUTAG.data, MUTAG.target
    #
    # # Splits the dataset into a training and a test set
    # G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.1, random_state=42)
    #
    # # Uses the shortest path kernel to generate the kernel matrices
    # gk = ShortestPath(normalize=True)
    # K_train = gk.fit_transform(G_train)
    # K_test = gk.transform(G_test)
    #
    # # Uses the SVM classifier to perform classification
    # clf = SVC(kernel="precomputed")
    # clf.fit(K_train, y_train)
    # y_pred = clf.predict(K_test)
    #
    # # Computes and prints the classification accuracy
    # acc = accuracy_score(y_test, y_pred)
    # print("Accuracy:", str(round(acc * 100, 2)) + "%")
#     --------------------------------------------------------------
#     ENZYMES_attr = fetch_dataset("ENZYMES", prefer_attr_nodes=True, verbose=False)
#     G, y = ENZYMES_attr.data, ENZYMES_attr.target
#
#     # Splits the dataset into a training and a test set
#     G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.1, random_state=42)
#
#     # Uses the graphhopper kernel to generate the kernel matrices
#     gk = PropagationAttr(normalize=False)
#     K_train = gk.fit_transform(G_train)
#     K_test = gk.transform(G_test)
#
#     # Uses the SVM classifier to perform classification
#     clf = SVC(kernel="precomputed")
#     clf.fit(K_train, y_train)
#     y_pred = clf.predict(K_test)
#
#     # Computes and prints the classification accuracy
#     acc = accuracy_score(y_test, y_pred)
#     print("Accuracy:", str(round(acc * 100, 2)) + "%")
if __name__ == '__main__':
    main()