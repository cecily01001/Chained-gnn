import click
import torch
from torch import nn
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from scapy.all import *
from scapy.error import Scapy_Exception
from scapy.layers.inet import IP, UDP, TCP
from scapy.layers.l2 import Ether
from scapy.packet import Padding
from scipy import sparse
from utils.utilsformalware import should_omit_packet, read_pcap, PREFIX_TO_Malware_ID
from utils.utilsforapps import PREFIX_TO_APP_ID
from utils.utilsforentropy import PREFIX_TO_ENTROPY_ID, ID_TO_ENTROPY

PcapFileLength = 200


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

def packet_to_sparse_array(packet, max_length=1500):
    arr = np.frombuffer(raw(packet), dtype=np.uint8)[0: max_length]
    # arr = np.frombuffer(raw(packet), dtype=np.uint8)[0: max_length] / 255
    if len(arr) < max_length:
        pad_width = max_length - len(arr)
        arr = np.pad(arr, pad_width=(0, pad_width), constant_values=0)
    arr = sparse.csr_matrix(arr)
    return arr

def transform_packet(packet, max_length):
    if should_omit_packet(packet):
        return None
    packet = remove_ether_header(packet)
    packet = pad_udp(packet)
    packet = mask_ip(packet)
    arr = packet_to_sparse_array(packet, max_length)
    return arr

def transform_pcap(pcap_path, pcapFeatureList, listofkind, kind):
    feature_matrix = []
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
    for i, packet in enumerate(read_pcap(pcap_path)):
        arr = transform_packet(packet, max_length)
        if arr is not None:
            if (node_num < PcapFileLength):
                node_num = node_num + 1
                feature_matrix.append(arr.todense().tolist()[0])
    while (node_num < PcapFileLength):
        feature_matrix.append([0] * 1500)
        node_num += 1
    label = label_dic.get(pcap_path.name.split('.')[0])
    listofkind.append(label)
    pcapFeatureList.append(feature_matrix)

class MyDataset(torch.utils.data.Dataset):  # ?????????????????????MyDataset,?????????????????????torch.utils.data.Dataset
    def __init__(self, pcapList, labelList, num_classes):  # ????????????????????????????????????
        self.pcapList = pcapList
        self.labelList = labelList
        self.num_classes = num_classes

    def __getitem__(self, index):  # ????????????????????????????????????????????????????????????????????????????????????
        return self.pcapList[index], self.labelList[index]  # return????????????return????????????????????????????????????????????????????????????batch??????????????????????????????

    def __len__(self):  # ???????????????????????????????????????????????????????????????????????????????????????????????????loader??????????????????
        return len(self.pcapList)


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.cnn=nn.Sequential(
            nn.Conv1d(
                in_channels=1500,
                out_channels=512,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU()
        )
        self.rnn = nn.LSTM(  # LSTM ???????????? nn.RNN() ?????????
            input_size=512,  # ??????????????????????????????
            hidden_size=512,  # rnn hidden unit
            num_layers=1,  # ????????? RNN layers
            batch_first=True,  # input & output ????????? batch size ??????????????????????????? e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(512, 41)  # ?????????

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM ????????? hidden states, h_n ?????????, h_c ?????????
        # h_c shape (n_layers, batch, hidden_size)
        # x = x.view(32, 200, 1500)
        # x=x.permute(1, 0, 2)
        # print(x.shape)
        x = x.transpose(2, 1)
        x=self.cnn(x).transpose(2, 1)
        print(x.shape)
        r_out, (h_n, h_c) = self.rnn(x, None)  # None ?????? hidden state ?????????0??? state
        # ?????????????????????????????? r_out ??????
        # ?????? r_out[:, -1, :] ???????????? h_n ??????
        out = self.out(h_n)
        out = torch.squeeze(out)
        return out

    def init_hidden(self):
        return torch.zeros(32, 512)


def collate_fn(batch):
    #  batch??????????????????????????????????????????????????????????????????dataset???_getitem__?????????
    batch = list(zip(*batch))
    labels = batch[1]
    texts = batch[0]
    del batch
    return texts, labels


def rnn_train(pcapDir, kind):
    data_dir_path = Path(pcapDir)
    if kind == 'app':
        label_dic = PREFIX_TO_APP_ID
    elif kind == 'malware':
        label_dic = PREFIX_TO_Malware_ID
    else:
        label_dic = PREFIX_TO_ENTROPY_ID
    num_classes = len(label_dic)
    pcapFeatureList = []
    listofkind = []
    for pcap_path in sorted(data_dir_path.iterdir()):
        transform_pcap(pcap_path, pcapFeatureList, listofkind, kind)

    train_g = []
    train_l = []
    valid_g = []
    valid_l = []
    test_g = []
    test_l = []
    temp_g = []
    temp_l = []

    for i in range(len(pcapFeatureList)):
        if i % 10 == 0:
            valid_g.append(pcapFeatureList[i])
            valid_l.append(listofkind[i])
        else:
            temp_g.append(pcapFeatureList[i])
            temp_l.append(listofkind[i])
    # for i in range(len(temp_g)):
    #     if i % 9 == 0:
    #         test_g.append(temp_g[i])
    #         test_l.append(temp_l[i])
    #     else:
    #         train_g.append(temp_g[i])
    #         train_l.append(temp_l[i])

    trainset = MyDataset(temp_g, temp_l, num_classes)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True,collate_fn=collate_fn)
    # print(len(trainloader))
    # print(trainloader)
    # ?????????
    # graphlist = []
    # listofkind = []
    # for pcap_path in sorted(valid_data_dir_path.iterdir()):
    #     transform_pcap(pcap_path, graphlist, listofkind,kind)
    # validset = MyDataset(valid_g, valid_l, num_classes)
    # validloader = DataLoader(validset, batch_size=32, shuffle=True)

    # ?????????
    # graphlist = []
    # listofkind = []
    # for pcap_path in sorted(testdata_dir_path.iterdir()):
    #     transform_pcap(pcap_path, graphlist, listofkind,kind)

    testset = MyDataset(valid_g, valid_l, num_classes)
    testloader = DataLoader(testset,batch_size=len(testset),shuffle=False,collate_fn=collate_fn)

    loss_func = nn.CrossEntropyLoss()
    classifier = RNN()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    for epoch in range(1, 300 + 1):
        # Train cycle
        total_loss = 0
        for i, (pcaps, labels) in enumerate(trainloader, 1):  # ?????????1????????? i ???1?????????
            # make_tensors???????????????????????????????????? ????????????????????????????????????
            pcaps = torch.tensor(pcaps, dtype=torch.float)
            labels=torch.tensor(labels, dtype=torch.long)
            output = classifier(pcaps)
            loss = loss_func(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f'Epoch: {epoch} ', end='')
            print(f'[{i * len(pcaps)}/{len(trainset)}] ', end='')
            print(f'loss={total_loss / (i * len(pcaps))}')

    correct = 0
    total = len(testset)
    with torch.no_grad():
        for i, (pcaps, labels) in enumerate(testloader, 1):
            pcaps = torch.tensor(pcaps, dtype=torch.float)
            labels = torch.tensor(labels, dtype=torch.long)
            output = classifier(pcaps)  # ??????
            pred = output.max(dim=1, keepdim=True)[1]  # ??????
            correct += pred.eq(labels.view_as(pred)).sum().item()  # ????????????????????????

        percent = '%.2f' % (100 * correct / total)
        print(f'Test set: Accuracy {correct}/{total} {percent}%')

@click.command()
@click.option('-k', '--kind', help='object to be classificated', required=True)
def main(kind):
    if kind == 'app':
        rnn_train("Dataset/Splite_Session/app", 'app')
    elif kind == 'malware':
        rnn_train("Dataset/Splite_Session/malware", 'malware')
    else:
        rnn_train("Dataset/Splite_Session/encrypted", 'entropy')

if __name__ == '__main__':
    main()
