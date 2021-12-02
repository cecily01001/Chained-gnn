import jieba
from torchtext.legacy import data
import re
from torchtext.vocab import Vectors
import dill
import torch

def tokenizer(text): # create a tokenizer function
    regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')
    text = regex.sub(' ', text)
    return [word for word in jieba.cut(text) if word.strip()]


def load_data(args):
    # print('加载数据中...')
    '''
    如果需要设置文本的长度，则设置fix_length,否则torchtext自动将文本长度处理为最大样本长度
    text = data.Field(sequential=True, tokenize=tokenizer, fix_length=args.max_len, stop_words=stop_words)
    '''
    text = data.Field(sequential=True, lower=True, tokenize=tokenizer)
    label = data.Field(sequential=False)

    text.tokenize = tokenizer
    train = data.TabularDataset(
            path='sampletrain.csv',
            # skip_header=True,
            # train='sampletrain.tsv',
            format='csv',
            fields=[('label', label), ('text', text)],
        )
    print("load vocab data")
    text.build_vocab(train)
    label.build_vocab(train)
    with open("data/pcap.vector", 'wb')as f:
        dill.dump(text.vocab, f)
    with open("data/label.vector", 'wb')as f:
        dill.dump(label.vocab, f)
    f1 = open("data/pcap.vector", 'rb')
    f2 = open("data/label.vector", 'rb')
    pcap = dill.load(f1)
    pcap_label = dill.load(f2)
    text.build_vocab(train)  # 此处改为你自己的词向量
    text.vocab= pcap

    # text.vocab.vectors=temp
    label.build_vocab(train)
    label.vocab=pcap_label
    # print(text.vocab.stoi)
    # for i in label.vocab:
    #     print(i)
    # args.embedding_dim = text.vocab.vectors.size()[-1]
    args.embedding_dim = 128
    args.vectors = text.vocab.vectors
    #
    # else:
    #     text.build_vocab(train, val)
    #     label.build_vocab(train, val)
    #     with open("data/pcap.vector", 'wb')as f:
    #         dill.dump(text.vocab, f)
    #     with open("data/label.vector", 'wb')as f:
    #         dill.dump(label.vocab, f)

    # print(text.vocab.itos)
    print()
    train_iter = data.Iterator(
            train,
            sort_key=lambda x: len(x.text),
            # batch_size=len(train), # 训练集设置batch_size,验证集整个集合用于测试
            batch_size=128,  # 训练集设置batch_size,验证集整个集合用于测试
            device=torch.device('cuda')
    )
    args.vocab_size = len(text.vocab)
    args.label_num = len(label.vocab)
    return train_iter