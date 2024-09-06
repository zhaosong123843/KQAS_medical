import jieba
import torch
import torchtext
import random
import numpy as np

from build_NLU_model.textCNN_model import TextCNN
from build_NLU_model.NLU_model_config import *

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def cut(sentence):
    stopwords_path = './cache/hit_stopwords.txt'
    stopwords = open(stopwords_path, encoding='utf-8').read().split('\n')
    return [token for token in jieba.lcut(sentence) if token not in stopwords]

TEXT = torchtext.legacy.data.Field(sequential=True, lower=True, tokenize=cut, init_token='<sos>', eos_token='<eos>',
                                   pad_token='<pad>', unk_token='<unk>')
LABEL = torchtext.legacy.data.LabelField(sequential=False, dtype=torch.long)

TEXT.vocab = torch.load('./cache/text_vocab.pt')
LABEL.vocab = torch.load('./cache/label_vocab.pt')


def nlu_predict(text):
    # set_seed()
    sample_text = text
    processed_text = TEXT.preprocess(sample_text)
    numericalized_text = TEXT.numericalize([processed_text])

    class_num = len(LABEL.vocab)  # 类别数目
    embedding_dim = TEXT.vocab.vectors.size()[-1]  # 词向量维度
    vectors = TEXT.vocab.vectors  # 词向量
    vocab_size = len(TEXT.vocab)  # 词表大小

    model = TextCNN(class_num=class_num,
                    filter_sizes=filter_size,
                    filter_num=filter_num,
                    vocabulary_size=vocab_size,
                    embedding_dimension=embedding_dim,
                    vectors=vectors,
                    dropout=dropout)

    model.load_state_dict(torch.load('./cache/nlu_best_model.bin'), False)
    model.eval()

    with torch.no_grad():
        output = model(numericalized_text)
        pre = output.argmax(dim=1).item()

    pre_label = LABEL.vocab.itos[pre]

    return pre_label


if __name__ == '__main__':
    text = '什么是牙龈肿痛？'
    print(nlu_predict(text))
