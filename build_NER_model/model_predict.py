import torch
import math

from torch.utils import data
from itertools import chain
from tqdm import tqdm
from sklearn import metrics

from build_NER_model.biLSTMCRF_data_process import Mydataset
from build_NER_model.biLSTMCRF_data_process import get_vocab_label
from build_NER_model.BiLSTM_CRF import BiLSTM_CRF
from build_NER_model.NER_model_config import *

# 从预测的标签中提取出实体
def entity_extract(pred):
    if not pred:
        return []

    cur_entity = None
    res = []
    start_idx,end_idx = 0,0

    for i,pre_single in enumerate(pred):
        pred_start_B = pre_single.startswith('B') # 实体标签开头B
        pred_entity = pre_single.split('_')[-1]

        if cur_entity:
            if pred_start_B or cur_entity != pred_entity:
                res.append({
                    'st_idx':start_idx,
                    'end_idx':i,
                    'label':cur_entity
                })
                cur_entity = None
        if pred_start_B:
            start_idx = i
            cur_entity = pred_entity
    if cur_entity:
        res.append({
            'st_idx':start_idx,
            'end_idx':len(pred),
            'label':cur_entity,
        })
    return res

path = 'label_test.json'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# embedding_size = 256
# hidden_dim = 256
# batch_size = 16

# 加载词表和实体标签
vocab,_,label_map,label_map_inv = get_vocab_label()

# 创建模型并且加载模型参数
model = BiLSTM_CRF(embedding_dim=embedding_size,hidden_dim=hidden_dim,vocab=vocab,label_map=label_map,device=device)
model.load_state_dict(torch.load('./cache/ner_best_model.bin'), strict=False)
model.to(device)

# 单句预测
def single_text_pre(test_text):
    # 单句预测要对句子进行拆分，拆分成训练时的batch_size

    model.eval()
    model.state = 'pred'

    text = [vocab.get(t, vocab['UNK']) for t in test_text]
    if len(text) < batch_size:
        text.extend([vocab['PAD']]*(batch_size-len(text)))

    seq_len = torch.tensor(len(text), dtype=torch.long).unsqueeze(0)
    seq_len.to(device)
    text = torch.tensor(text, dtype=torch.long).unsqueeze(0)
    text = text.to(device)
    batch_tag = model(text, seq_len)
    pre = [label_map_inv[t] for t in batch_tag]

    return pre

# 测试集预测
def test_pre():

    test_path = './data/train.txt'
    test_dataset = Mydataset(test_path, batch_size)
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                       collate_fn=test_dataset.collect_fn)

    all_label = []
    all_pred = []
    model.eval()
    model.state = 'eval'
    with torch.no_grad():
        for text, label, seq_len in tqdm(test_dataloader, desc='eval: '):
            text = text.to(device)
            print(text)
            seq_len = seq_len.to(device)
            batch_tag = model(text, seq_len, label)
            all_label.extend(
                [[label_map_inv[t] for t in l[:seq_len[i]].tolist()] for i, l in enumerate(label)])
            all_pred.extend([[label_map_inv[t] for t in l] for l in batch_tag])

    all_label = list(chain.from_iterable(all_label))
    all_pred = list(chain.from_iterable(all_pred))
    sort_labels = [k for k in label_map.keys()]

    # 使用sklearn库得到F1分数
    # f1 = metrics.f1_score(all_label, all_pred, average='macro', labels=sort_labels[:-3])
    print(all_label[:20])
    print('*'*10)
    print(all_pred[:20])
    print(metrics.classification_report(
        all_label, all_pred, labels=sort_labels[:-3], digits=3
    ))

def ner_predict(text):

    test_text = text
    text_len = len(test_text)
    pre = []
    if text_len > batch_size:
        num = math.ceil(text_len / batch_size)
        for i in range(num):
            text=test_text[i*batch_size:(i+1)*batch_size]
            pre.extend(single_text_pre(text))
    else:
        pre = single_text_pre(test_text)
    res = entity_extract(pre)

    entity = []
    for item in res:
        start_index = item['st_idx']
        end_index = item['end_idx']
        entity.append([test_text[start_index:end_index],item['label']])

    return entity

if __name__ == "__main__":

    # path = './data/train.txt'
    # with open(path,'r',encoding='utf-8') as file:
    #     lines = file.readlines()
    #
    # labels = []
    # texts = ''
    # for item in lines:
    #     item = item.split(' ')
    #     if len(item) >= 2:
    #         texts += item[0]
    #         labels.append(item[1].strip('\n'))
    #     if len(texts) > 100:
    #         break

    test_text = '便秘俩个月了是怎么回事'
    text_len = len(test_text)
    pre = []
    if text_len > batch_size:
        num = math.ceil(text_len / batch_size)
        for i in range(num):
            text=test_text[i*batch_size:(i+1)*batch_size]
            pre.extend(single_text_pre(text))
    else:
        pre = single_text_pre(test_text)
    print(pre)
    res = entity_extract(pre)
    print(res)

    entity = []
    for item in res:
        start_index = item['st_idx']
        end_index = item['end_idx']
        entity.append([test_text[start_index:end_index+1],item['label']])

    print(entity)

