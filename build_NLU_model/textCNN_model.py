import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.metrics import classification_report
from build_NLU_model.process import DataProcess
from build_NLU_model.NLU_model_config import *

import warnings
warnings.filterwarnings('ignore')

# 定义模型保存函数
def save(model, save_dir, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = 'bestmodel_steps{}.pt'.format(steps)
    save_bestmodel_path = os.path.join(save_dir, save_path)
    torch.save(model.state_dict(), save_bestmodel_path)


def dev_eval(dev_iter, model):
    model.eval()
    corrects, avg_loss = 0, 0
    all_pre = []
    all_labels = []
    with torch.no_grad():
        for batch in dev_iter:
            feature, target = batch.text, batch.label

            if torch.cuda.is_available():
                feature, target = feature.cuda(), target.cuda()
            logits = model(feature)
            loss = F.cross_entropy(logits, target)
            avg_loss += loss.item()

            pre = torch.max(logits, 1)[1].view(target.size()).data
            corrects += (pre == target.data).sum()

            all_pre.extend(pre.cpu().numpy())
            all_labels.extend(target.data.cpu().numpy())

    size = len(dev_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))

    # 打印分类报告
    print("Classification Report:")
    print(classification_report(all_labels, all_pre, digits=3))

    return accuracy


def train(train_iter, dev_iter, model):
    if torch.cuda.is_available():  # 判断是否有GPU，如果有把模型放在GPU上训练，速度质的飞跃
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 梯度下降优化器，采用Adam
    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, epochs + 1):
        print(f'---------------------eopch:{epoch}-----------------------------')
        for batch in tqdm(train_iter):
            feature, target = batch.text, batch.label
            # print(feature, target)
            # exit()
            if torch.cuda.is_available():  # 如果有GPU将特征更新放在GPU上
                feature, target = feature.cuda(), target.cuda()
            optimizer.zero_grad()  # 将梯度初始化为0，每个batch都是独立训练地，因为每训练一个batch都需要将梯度归零
            logits = model(feature)
            loss = F.cross_entropy(logits, target)  # 计算损失函数 采用交叉熵损失函数
            loss.backward()  # 反向传播
            optimizer.step()  # 放在loss.backward()后进行参数的更新
            optimizer.zero_grad()
            steps += 1

        dev_acc = dev_eval(dev_iter, model)
        if dev_acc > best_acc:
            best_acc = dev_acc
            # last_step = steps
            print('Saving best model, acc: {:.4f}%\n'.format(best_acc))
            torch.save(model.state_dict(), save_dir)
                # else:
                #     if steps - last_step >= early_stopping:
                #         print('\n提前停止于 {} steps, acc: {:.4f}%'.format(last_step, best_acc))
                #         raise KeyboardInterrupt

class TextCNN(nn.Module):
    def __init__(self,
                 class_num,  # 最后输出的种类数
                 filter_sizes,  # 卷积核的长也就是滑动窗口的长
                 filter_num,  # 卷积核的数量
                 vocabulary_size,  # 词表的大小
                 embedding_dimension,  # 词向量的维度
                 vectors,  # 词向量
                 dropout):  # dropout率
        super(TextCNN, self).__init__()  # 继承nn.Module

        chanel_num = 1  # 通道数，也就是一篇文章一个样本只相当于一个feature map

        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)  # 嵌入层
        self.embedding = self.embedding.from_pretrained(vectors)  # 嵌入层加载预训练词向量

        self.convs = nn.ModuleList(
            [nn.Conv2d(chanel_num, filter_num, (fsz, embedding_dimension)) for fsz in filter_sizes])  # 卷积层
        self.dropout = nn.Dropout(dropout)  # dropout
        self.fc = nn.Linear(len(filter_sizes) * filter_num, class_num)  # 全连接层

    def forward(self, x):
        # x维度[句子长度,一个batch中所包含的样本数] 例:[3451,128]
        x = self.embedding(x)  # 经过嵌入层之后x的维度，[一个batch中所包含的样本数,句子长度,词向量维度] 例：[128,3451,300]
        x = x.permute(1, 0, 2)  # permute函数将样本数和句子长度换一下位置，[一个batch中所包含的样本数,句子长度,词向量维度] 例：[128,3451,300]
        x = x.unsqueeze(1)  # # conv2d需要输入的是一个四维数据，所以新增一维feature map数 unsqueeze(1)表示在第一维处新增一维，[一个batch中所包含的样本数,一个样本中的feature map数，句子长度,词向量维度] 例：[128,1,3451,300]

        # 添加填充
        max_filter_size = max(self.convs, key=lambda conv: conv.kernel_size[0]).kernel_size[0]
        pad_size = max_filter_size - 1
        x = F.pad(x, (0, 0, pad_size, pad_size), mode='constant', value=0)

        x = [conv(x) for conv in self.convs]  # 与卷积核进行卷积
        x = [sub_x.squeeze(3) for sub_x in x]  # squeeze(3)判断第三维是否是1，如果是则压缩
        x = [F.relu(sub_x) for sub_x in x]  # ReLU激活函数激活
        x = [F.max_pool1d(sub_x, sub_x.size(2)) for sub_x in x]  # 池化层
        x = [sub_x.squeeze(2) for sub_x in x]  # 判断第二维是否为1，若是则压缩
        x = torch.cat(x, 1)  # 进行拼接
        x = self.dropout(x)  # 去除掉一些神经元防止过拟合
        logits = self.fc(x)  # 全接连层

        return logits


if __name__ == "__main__":

    dp = DataProcess(batch_size)

    LABEL = dp.LABEL
    TEXT = dp.TEXT

    train_iter = dp.train_iter
    dev_iter = dp.val_iter

    class_num = len(LABEL.vocab)  # 类别数目
    vocab_size = len(TEXT.vocab)  # 词表大小
    embedding_dim = TEXT.vocab.vectors.size()[-1]  # 词向量维度
    vectors = TEXT.vocab.vectors  # 词向量

    textcnn_model = TextCNN(class_num=class_num,
                            filter_sizes=filter_size,
                            filter_num=filter_num,
                            vocabulary_size=vocab_size,
                            embedding_dimension=embedding_dim,
                            vectors=vectors,
                            dropout=dropout)

    print(10 * '*' + '开始训练' + '*' * 10)

    train(train_iter=train_iter, dev_iter=dev_iter, model=textcnn_model)
