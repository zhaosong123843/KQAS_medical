import os
import time
import jieba
import torch
import torchtext
import pandas as pd
import numpy as np

class DataProcess:

    def __init__(self,batch_size):

        # 停用词
        stopwords_path = './cache/hit_stopwords.txt'
        self.stopwords = open(stopwords_path,encoding='utf-8').read().split('\n')

        print('-' * 10 + '停用词加载完成' + '-' * 10)

        # 加载预训练词向量
        self.pre_trained_name = 'sgns.sogou.word'
        self.pre_trained_path = '.\\cache'
        self.vectors = torchtext.vocab.Vectors(name=self.pre_trained_name, cache=self.pre_trained_path)

        print('-' * 10 + '预训练词向量加载完成' + '-' * 10)

        # 定义Field
        self.TEXT = torchtext.legacy.data.Field(sequential=True, lower=True, tokenize=self.cut,init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
        self.LABEL = torchtext.legacy.data.LabelField(sequential=False,dtype=torch.long)

        # 加载数据
        # start_time = time.time()
        fields = [('text', self.TEXT),('label', self.LABEL)]
        self.train_dataset, self.val_dataset = torchtext.legacy.data.TabularDataset.splits(
            path='.\data',
            format='csv',
            skip_header=True,  # 是否跳过表头
            train='self_train.csv',
            validation='self_test.csv',
            fields=fields,  # 定义数据对应的表头
        )

        # end_time = time.time()

        print('-' * 10 + '数据加载完成' + '-' * 10)
        # print('数据加载所用时间%s' % (end_time - start_time))

        # 创建词表
        if os.path.exists(',/cache//text_vocab.pt'):
            self.TEXT.vocab = torch.load('./cache/text_vocab.pt')
            self.LABEL.vocab = torch.load('./cache/label_vocab.pt')
            print('-' * 10 + '词表加载完成' + '-' * 10)
        else:
            self.TEXT.build_vocab(self.train_dataset, self.val_dataset, vectors=self.vectors, min_freq=2)
            self.LABEL.build_vocab(self.train_dataset, self.val_dataset)
            print('-' * 10 + '词表创建完成' + '-' * 10)

            torch.save(self.TEXT.vocab, './cache/text_vocab.pt')
            torch.save(self.LABEL.vocab, './cache/label_vocab.pt')

        # 重新创建迭代器
        self.train_iter, self.val_iter = torchtext.legacy.data.BucketIterator.splits(
            (self.train_dataset, self.val_dataset),
            batch_sizes=(batch_size, batch_size, batch_size),
            sort_key=lambda x: len(x.text)
        )

    def cut(self, sentence):
        return [token for token in jieba.lcut(sentence) if token not in self.stopwords]

