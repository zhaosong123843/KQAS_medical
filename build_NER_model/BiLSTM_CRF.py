import torch
import torch.nn as nn
import torch.nn.functional as F


# log sum exp 增强数值稳定性
# 改进了torch版本原始函数.可适用于两种情况计算得分
def log_sum_exp(vec):
    max_score, _ = torch.max(vec, dim=-1)
    max_score_broadcast = max_score.unsqueeze(-1).repeat_interleave(vec.shape[-1], dim=-1)
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=-1))


class CRF(nn.Module):
    def __init__(self, label_map, device='cpu'):
        super(CRF, self).__init__()
        self.label_map = label_map
        self.label_map_inv = {v: k for k, v in label_map.items()}
        self.tagset_size = len(self.label_map)
        self.device = device

        # 转移概率矩阵
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # 增加开始和结束标志，并手动干预转移概率
        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"
        self.transitions.data[self.label_map[self.START_TAG], :] = -10000
        self.transitions.data[:, self.label_map[self.STOP_TAG]] = -10000

    def _forward_alg(self, feats, seq_len):
        # 手动设置初始得分，让开始标志到其他标签的得分最高
        init_alphas = torch.full((self.tagset_size,), -10000.)
        init_alphas[self.label_map[self.START_TAG]] = 0.

        # 记录所有时间步的得分，为了解决序列长度不同问题，后面直接取各自长度索引的得分即可
        # shape：(batch_size, seq_len + 1, tagset_size)
        forward_var = torch.zeros(feats.shape[0], feats.shape[1] + 1, feats.shape[2], dtype=torch.float32,
                                  device=self.device)
        forward_var[:, 0, :] = init_alphas

        # 将转移概率矩阵复制 batch_size 次，批次内一起进行计算，矩阵计算优化，加快运行效率
        # shape：(batch_size, tagset_size) -> (batch_size, tagset_size, tagset_size)
        transitions = self.transitions.unsqueeze(0).repeat(feats.shape[0], 1, 1)
        # 对所有时间步进行遍历
        for seq_i in range(feats.shape[1]):
            # 取出当前词发射概率
            emit_score = feats[:, seq_i, :]
            # 前一时间步得分 + 转移概率 + 当前时间步发射概率
            tag_var = (
                    forward_var[:, seq_i, :].unsqueeze(1).repeat(1, feats.shape[2],
                                                                 1)  # (batch_size, tagset_size, tagset_size)
                    + transitions
                    + emit_score.unsqueeze(2).repeat(1, 1, feats.shape[2])
            )

            # max_scores, _ = torch.max(tag_var, dim=2)
            # forward_var[:, seq_i + 1, :] = max_scores + torch.logsumexp(tag_var - max_scores.unsqueeze(2), dim=2)
            # 这里必须调用clone，不能直接在forward_var上修改，否则在梯度回传时会报错
            cloned = forward_var.clone()
            cloned[:, seq_i + 1, :] = log_sum_exp(tag_var)
            forward_var = cloned

        # 按照不同序列长度不同取出最终得分
        forward_var = forward_var[range(feats.shape[0]), seq_len, :]
        # 手动干预,加上结束标志位的转移概率
        terminal_var = forward_var + self.transitions[self.label_map[self.STOP_TAG]].unsqueeze(0).repeat(feats.shape[0],
                                                                                                         1)
        # 得到最终所有路径的分数和
        alpha = log_sum_exp(terminal_var)
        return alpha

    # 修改矩阵计算方式，加速计算
    def _score_sentence(self, feats, tags, seq_len):
        # 初始化,大小为(batch_size,)
        score = torch.zeros(feats.shape[0], device=self.device)
        # 将开始标签拼接到序列上起始位置，参与分数计算
        start = torch.tensor([self.label_map[self.START_TAG]], device=self.device).unsqueeze(0).repeat(feats.shape[0],
                                                                                                       1)
        tags = torch.cat([start, tags], dim=1)
        # 在batch上遍历
        for batch_i in range(feats.shape[0]):
            # 采用矩阵计算方法，加快运行效率
            # 取出当前序列所有时间步的转移概率和发射概率进行相加，由于计算真实标签序列的得分，所以只选择标签的路径
            score[batch_i] = torch.sum(
                self.transitions[tags[batch_i, 1:seq_len[batch_i] + 1], tags[batch_i, :seq_len[batch_i]]]) \
                             + torch.sum(feats[batch_i, range(seq_len[batch_i]), tags[batch_i][1:seq_len[batch_i] + 1]])
            # 最后加上结束标志位的转移概率
            score[batch_i] += self.transitions[self.label_map[self.STOP_TAG], tags[batch_i][seq_len[batch_i]]]
        return score

    # 维特比算法得到最优路径,原始torch函数
    def _viterbi_decode(self, feats):
        backpointers = []

        # 手动设置初始得分，让开始标志到其他标签的得分最高
        init_vvars = torch.full((1, self.tagset_size), -10000., device=self.device)
        init_vvars[0][self.label_map[self.START_TAG]] = 0

        # 用于记录前一时间步的分数
        forward_var = init_vvars
        # 传入的就是单个序列,在每个时间步上遍历
        for feat in feats:
            # 将上一时间步的总概率复制tagset_size次，以便一次性加上所有转移概率
            forward_var = forward_var.repeat(feat.shape[0], 1)
            next_tag_var = forward_var + self.transitions
            # 对每个标签位置取最大值的索引
            bptrs_t = torch.max(next_tag_var, 1)[1].tolist()
            # 取出当前时间步所有最大值的概率
            viterbivars_t = next_tag_var[range(forward_var.shape[0]), bptrs_t]
            # 加上当前时间步的发射概率
            forward_var = (viterbivars_t + feat).view(1, -1)
            # 记录最大值的索引，后续回溯用
            backpointers.append(bptrs_t)

        # 手动加入转移到结束标签的概率
        terminal_var = forward_var + self.transitions[self.label_map[self.STOP_TAG]]
        # 在最终位置得到最高分数所对应的索引
        best_tag_id = torch.max(terminal_var, 1)[1].item()
        # 最高分数
        path_score = terminal_var[0][best_tag_id]

        # 回溯，向后遍历得到最优路径
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # 弹出开始标签
        start = best_path.pop()
        assert start == self.label_map[self.START_TAG]  # Sanity check
        # 将路径反转
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, feats, tags, seq_len):
        # 所有路径得分
        forward_score = self._forward_alg(feats, seq_len)
        # 标签路径得分
        gold_score = self._score_sentence(feats, tags, seq_len)
        # 返回 batch 分数的平均值
        return torch.mean(forward_score - gold_score)

class FocalLoss(nn.Module):
    def __init__(self,alpha=0.25,gamma=2.0,reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self,inputs,targets,mask):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs,targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            focal_loss = torch.mean(focal_loss * mask.float())
        elif self.reduction == 'sum':
            focal_loss = torch.sum(focal_loss * mask.float())

        return focal_loss

class BiLSTM_CRF(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab, label_map, device='cpu'):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim  # 词向量维度
        self.hidden_dim = hidden_dim  # 隐层维度
        self.vocab_size = len(vocab)  # 词表大小
        self.tagset_size = len(label_map)  # 标签个数
        self.device = device
        # 记录状态，'train'、'eval'、'pred'对应三种不同的操作
        self.state = 'train'  # 'train'、'eval'、'pred'

        self.word_embeds = nn.Embedding(self.vocab_size, embedding_dim)
        # BiLSTM会将两个方向的输出拼接，维度会乘2，所以在初始化时维度要除2
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=2, bidirectional=True, batch_first=True)

        # BiLSTM 输出转化为各个标签的概率，此为CRF的发射概率
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size, bias=True)
        # 初始化CRF类
        self.crf = CRF(label_map, device)
        self.dropout = nn.Dropout(p=0.5, inplace=True)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)

        self.focal_loss = FocalLoss()

    def _get_lstm_features(self, sentence, seq_len):
        embeds = self.word_embeds(sentence)
        self.dropout(embeds)

        packed = torch.nn.utils.rnn.pack_padded_sequence(embeds, seq_len, enforce_sorted=False, batch_first=True)
        lstm_out, _ = self.lstm(packed)
        seq_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        seqence_output = self.layer_norm(seq_unpacked)
        lstm_feats = self.hidden2tag(seqence_output)
        return lstm_feats

    def forward(self, sentence, seq_len, tags='',mask=None):

        if mask == None:
            mask = torch.ones_like(sentence,dtype=torch.bool,device=self.device)

        # 输入序列经过BiLSTM得到发射概率
        feats = self._get_lstm_features(sentence, seq_len)

        # 根据 state 判断哪种状态，从而选择计算损失还是维特比得到预测序列
        if self.state == 'train':
            tags = tags.long()
            # focal_loss = self.focal_loss(feats.view(-1,self.tagset_size),tags.view(-1),mask.view(-1))
            crf_loss = self.crf.neg_log_likelihood(feats, tags, seq_len)
            # loss = crf_loss + focal_loss

            return crf_loss
        elif self.state == 'eval':
            all_tag = []
            for i, feat in enumerate(feats):
                all_tag.append(self.crf._viterbi_decode(feat[:seq_len[i]])[1])
                # 应用 softmax 函数得到概率分布
            # probs = F.softmax(feats, dim=-1)
            # 使用 argmax 获取最大概率对应的索引作为预测标签
            # _, predicted_tags = torch.max(probs, dim=-1)
            # 重新调整形状以适应序列长度
            # predicted_tags = predicted_tags.view(sentence.size(0), -1)

            return all_tag
        else:
            return self.crf._viterbi_decode(feats[0])[1]
