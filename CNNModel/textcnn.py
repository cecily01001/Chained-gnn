import csv
from pathlib import Path
import numpy as np
from scapy.all import *
from scapy.layers.inet import IP, UDP,TCP
from scapy.layers.l2 import Ether
from scapy.packet import Padding
from scipy import sparse
from utilsforapps import PREFIX_TO_APP_ID


f = open(r'sampletrain.csv', 'w', encoding='utf-8', newline="")
tsv_w = csv.writer(f)
tsv_w.writerow(['label', 'text'])
def remove_ether_header(packet):
    if Ether in packet:
        return packet[Ether].payload

    return packet
def mask_ip(packet):
    if IP in packet:
        packet[IP].src = '0.0.0.0'
        packet[IP].dst = '0.0.0.0'

    return packet
def read_pcap(path: Path):
    packets = rdpcap(str(path))

    return packets
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

def packet_to_sparse_array(packet,max_length=1500):
    arr = np.frombuffer(raw(packet), dtype=np.uint8)[0:max_length]
    leng=len(arr)
    if leng < max_length:
        pad_width = max_length - leng
        arr = np.pad(arr, pad_width=(0, pad_width), constant_values=0)
    # print(arr)
    arr = sparse.csr_matrix(arr)
    return arr

def transform_packet(packet):
    packet = remove_ether_header(packet)
    packet = pad_udp(packet)
    packet = mask_ip(packet)
    arr = packet_to_sparse_array(packet)
    return arr

def transform_pcap(path):
    j = 0
    for i, packet in enumerate(read_pcap(path)):
        j = j + 1
        if j%10==0:
            arr = transform_packet(packet)
            if arr is not None:
                prefix = path.name.split('.')[0]
                app_label = PREFIX_TO_APP_ID.get(prefix)
            tsv_w.writerow([app_label, raw(packet)])


def make_app_graph(data_dir_path, valid_data_dir_path, testdata_dir_path):
    testdata_dir_path = Path(testdata_dir_path)
    valid_data_dir_path = Path(valid_data_dir_path)
    data_dir_path = Path(data_dir_path)
    for pcap_path in sorted(data_dir_path.iterdir()):
        transform_pcap(pcap_path)
    for pcap_path in sorted(testdata_dir_path.iterdir()):
        transform_pcap(pcap_path)
    for pcap_path in sorted(valid_data_dir_path.iterdir()):
        transform_pcap(pcap_path)
make_app_graph("../Dataset/Splite_Session/app/Train", "../Dataset/Splite_Session/app/valid", "../Dataset/Splite_Session/app/Test")
# make_app_graph("C:/Users/cecil/Desktop/pcapfiles", "C:/Users/cecil/Desktop/pcapfiles", "C:/Users/cecil/Desktop/pcapfiles")