import warnings
warnings.filterwarnings('ignore')

import time
import torch
import datetime
import torch.optim as optim

from torch.utils import data
from sklearn import metrics
from itertools import chain
from tqdm import tqdm

from build_NER_model.NER_model_config import *
from build_NER_model.BiLSTM_CRF import BiLSTM_CRF
from build_NER_model.biLSTMCRF_data_process import Mydataset



device = "cuda:0" if torch.cuda.is_available() else "cpu"

# train_path = './data/train.txt'
train_path = './data/train.txt'
train_dataset = Mydataset(train_path,batch_size)

val_path = './data/dev.txt'
val_dataset = Mydataset(val_path,batch_size)

# 创建 DataLoader 实例
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collect_fn)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=val_dataset.collect_fn)

model = BiLSTM_CRF(embedding_size, hidden_dim, train_dataset.vocab, train_dataset.label_map, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def train():
    total_start = time.time()
    best_score = 0
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        model.state = 'train'
        print('训练中............')
        for step, (text, label, seq_len) in enumerate(train_dataloader, start=1):

            # start = time.time()
            text = text.to(device)
            label = label.to(device)
            seq_len = seq_len.to(device)

            loss = model(text, seq_len, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # if steps % 10 == 0:  # 每训练多少步计算一次准确率，我这边是1，可以自己修改
            # corrects = (torch.max(loss, 1)[1].view(
            #     target.size()).data == target.data).sum()  # logits是[128,10],torch.max(logits, 1)也就是选出第一维中概率最大的值，输出为[128,1],torch.max(logits, 1)[1]相当于把每一个样本的预测输出取出来，然后通过view(target.size())平铺成和target一样的size (128,),然后把与target中相同的求和，统计预测正确的数量
            # train_acc = 100.0 * corrects / batch.batch_size  # 计算每个mini batch中的准确率
            # print('steps:{0} - loss: {1}  acc:{2}'.format(
            #     steps,
            #     loss.item(),
            #     train_acc))

            print(f'Epoch: [{epoch + 1}/{epochs}],'
                  f'  cur_epoch_finished: {step * batch_size / len(train_dataset) * 100:2.2f}%,'
                  f'  loss: {loss.item():2.4f},'
                  f'  cur_epoch_remaining_time: {datetime.timedelta(seconds=int((len(train_dataloader) - step) / step * (time.time() - epoch_start)))}',
                  f'  total_remaining_time: {datetime.timedelta(seconds=int((len(train_dataloader) * epochs - (len(train_dataloader) * epoch + step)) / (len(train_dataloader) * epoch + step) * (time.time() - total_start)))}')

        # 每周期验证一次，保存最优参数
        score = evaluate(model,val_dataset,val_dataloader)
        if score > best_score:
            print(f'score increase:{best_score} -> {score}')
            best_score = score
            torch.save(model.state_dict(), './cache/ner_best_model.bin')
        print(f'current best score: {best_score}')

        with open('./train_logging','a',encoding='utf-8') as file:
            file.write(f"epoch:{epoch}--------->score:{score}\n")


def evaluate(model,val_dataset,val_dataloader):
    # model.load_state_dict(torch.load('./cache/nlu_best_model.bin'))
    all_label = []
    all_pred = []
    model.eval()
    model.state = 'eval'
    with torch.no_grad():
        for text, label, seq_len in tqdm(val_dataloader, desc='eval: '):
            text = text.to(device)
            seq_len = seq_len.to(device)
            batch_tag = model(text,seq_len ,label)

            all_label.extend(
                [[val_dataset.label_map_inv[t] for t in l[:seq_len[i]].tolist()] for i, l in enumerate(label)])
            all_pred.extend([[val_dataset.label_map_inv[int(t)] for t in l] for l in batch_tag])

    all_label = list(chain.from_iterable(all_label))
    all_pred = list(chain.from_iterable(all_pred))
    sort_labels = [k for k in train_dataset.label_map.keys()]
    # print(len(all_label))
    # print(len(all_pred))
    # 使用sklearn库得到F1分数
    f1 = metrics.f1_score(all_label, all_pred, average='macro', labels=sort_labels[:-3])

    print(metrics.classification_report(
        all_label, all_pred, labels=sort_labels[:-3], digits=3
    ))
    return f1


if __name__ == "__main__":
    train()
    # test_model = BiLSTM_CRF(embedding_size, hidden_dim, train_dataset.vocab, train_dataset.label_map, device).to(device)
    # test_model.load_state_dict(torch.load('./cache/ner_best_model.bin'))
    # evaluate(test_model,train_dataset,train_dataloader)

