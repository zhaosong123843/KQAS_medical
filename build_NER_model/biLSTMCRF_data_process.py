import os
import jieba
import json
import pickle
import torch
import numpy as np
from torch.utils import data


# 建立词表和标签表
def get_vocab_label():
    # path = './data/train.txt'
    vocab_path = './cache/vocab.pkl'
    label_map_path = './cache/label_map.pkl'
    stopwords_path = './cache/hit_stopwords.txt'

    if os.path.exists(label_map_path):
        with open(label_map_path, 'r', encoding='utf-8') as file:
            label_map = json.load(file)

    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as fp:
            vocab = pickle.load(fp)
    else:
        train_path = './data/train.txt'
        test_path = './data/test.txt'
        dev_path = './data/dev.txt'

        lines = []
        with open(train_path, 'r', encoding='utf-8') as file:
            lines.extend(file.readlines())
        with open(test_path, 'r', encoding='utf-8') as file:
            lines.extend(file.readlines())
        with open(dev_path, 'r', encoding='utf-8') as file:
            lines.extend(file.readlines())

        stopwords = open(stopwords_path, encoding='utf-8').read().split('\n')
        # PAD:在一个batch中不同长度的序列用该字符补齐，padding
        # UNK:当验证集活测试机出现词表以外的词时，用该字符代替。unknown
        vocab = {'PAD': 0, 'UNK': 1}
        label_map = {}
        # 将字符存入词表
        for i in range(len(lines)):
            content = lines[i].split(' ')
            try:
                word = content[0]
                if word in stopwords:
                    continue
                if word not in vocab:
                    vocab[word] = len(vocab)

                label = content[1]
                if label.startswith(('B_','I_')):
                    label = label.split('\n')
                    try:
                        if label_map[label[0]]:
                            continue
                    except KeyError:
                        label_map[label[0]] = len(label_map)
            except:
                continue

        label_map['O'] = len(label_map)

        # 对于BiLSTM+CRF网络，需要增加开始和结束标签，以增强其标签约束能力
        START_TAG = "<START>"
        STOP_TAG = "<STOP>"
        label_map[START_TAG] = len(label_map)
        label_map[STOP_TAG] = len(label_map)

        with open(label_map_path, 'w', encoding='utf-8') as fp:
            json.dump(label_map, fp, indent=4)

        with open(vocab_path, 'wb') as fp:
            pickle.dump(vocab, fp)

    vocab_inv = {v: k for k, v in vocab.items()}
    label_map_inv = {v: k for k, v in label_map.items()}

    return vocab, vocab_inv, label_map,label_map_inv

def get_text_label(path):
    text_label_path = path

    stopwords_path = './cache/hit_stopwords.txt'
    stopwords = open(stopwords_path, encoding='utf-8').read().split('\n')

    texts = []
    label = []
    with open(text_label_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    for line in lines:
        content = line.split(' ')

        if content[0] == '\n':
            continue
        if content[0] in stopwords:
            continue
        else:
            texts.append(content[0])
            label.append(content[1])

    return texts,label


# 定义dataset类，dataset类是pytorch中用来处理数据的抽象类
class Mydataset(data.Dataset):
    def __init__(self,path,batch_size):

        self.path = path
        # 获取带BIO标注的数据
        self.data = get_text_label(path)

        # 去停用词

        # 获取实体标签
        # 获取词表
        self.vocab,self.vocab_inv,self.label_map,self.label_map_inv = get_vocab_label()
        self.batch_size = batch_size
        self.get_points()


    # getitem和len是dataset类中的方法，在处理自定义数据集时需要重写该方法
    def __getitem__(self, item):

        # 将data中的中文和label中的英文根据中文词表以及实体标签索引变为数字索引
        text = self.data[0][self.points[item]:self.points[item+1]]
        label = self.data[1][self.points[item]:self.points[item+1]]
        label = [s.replace("\n", "") for s in label]  # 删除换行符\n

        t = [self.vocab.get(t, self.vocab['UNK']) for t in text]
        l = [self.label_map[l] for l in label]

        return t,l

    def __len__(self):
        return len(self.points) - 1

    # 文本长度填充
    def collect_fn(self,batch):
        # print(batch)
        _text = []
        _label = []
        seq_len = []
        for _t,_l in batch:
            _text.append(_t)
            _label.append(_l)
            seq_len.append(len(_t))

        # 最大长度作为填充的标准
        max_len = max(seq_len)

        # 转化为tensor张量
        text = torch.tensor([t + [self.vocab['PAD']] * (max_len - len(t)) for t in _text],dtype=torch.long)
        label = torch.tensor([l + [self.label_map['O']] * (max_len - len(l)) for l in _label],dtype=torch.long)
        seq_len = torch.tensor(seq_len,dtype=torch.long)

        return text,label,seq_len

    # 文本分割
    def get_points(self):

        label = self.data[1]
        label = [s.replace("\n", "") for s in label]  # 删除换行符\n

        self.points = [0]
        i = 0
        while True:
            if i+self.batch_size >= len(label):
                self.points.append(len(label))
                break
            if label[i+self.batch_size] == 'O':
                i += self.batch_size
                self.points.append(i)
            else:
                i += 1


if __name__ == '__main__':

    # path = './data/train.txt'

    vocab = get_vocab_label()

    # get_label(path)
    # texts,label = get_text_label(path)

    # label = [s.replace("\n", "") for s in label]
    # texts = [s.replace("\n", "") for s in texts]
    import pandas as pd

    # stopwords_path = './cache/hit_stopwords.txt'
    # stopwords = open(stopwords_path, encoding='utf-8').read().split('\n')


    # with open('./new_test.txt','w',encoding='utf-8') as f:
    #     for i in range(len(texts)):
    #         token = texts[i]
    #         if token not in stopwords:
    #             content = token+' '+label[i]
    #             f.write(content)

    # with open(path,'r',encoding='utf-8') as file:
    #     lines = file.readlines()
    #
    # texts = []
    # labels = []
    #
    # for item in lines:
    #     item = item.split(' ')
    #     if len(item) >= 2:
    #         texts.append(item[0])
    #         labels.append(item[1])
    #
    # df = pd.DataFrame(columns=['texts', 'labels'])
    # df['texts'] = texts
    # df['labels'] = labels
    #
    # print(df['labels'].value_counts(normalize=True))
    #
    # with open('./new_test.txt','r',encoding='utf-8') as f:
    #     new_lines = f.readlines()
    #
    # new_texts = []
    # new_labels = []
    #
    # for item in new_lines:
    #     item = item.split(' ')
    #     if len(item) >= 2:
    #         new_texts.append(item[0])
    #         new_labels.append(item[1])
    #
    # new_df = pd.DataFrame(columns=['texts','labels'])
    # new_df['texts'] = new_texts
    # new_df['labels'] = new_labels
    #
    # print(new_df['labels'].value_counts(normalize=True))