import os
import numpy as np
import logging
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

logger = logging.getLogger(__name__)

bert_model = '/Users/admin/Desktop/bert-base-chinese-model'
tokenizer = BertTokenizer.from_pretrained(bert_model)
# VOCAB = ('<PAD>', 'O', 'I-LOC', 'B-PER', 'I-PER', 'I-ORG', 'B-LOC', 'B-ORG')
# VOCAB = ('<PAD>', '[CLS]', '[SEP]', 'O', 'B-INF', 'I-INF', 'B-PAT', 'I-PAT', 'B-OPS',
#         'I-OPS', 'B-DSE', 'I-DSE', 'B-DRG', 'I-DRG', 'B-LAB', 'I-LAB')
# 根据自己数据集打的标签来修改VOCAB,前三个是Bert对应的标签，依次填充、句首、句尾分隔标签。
VOCAB =('<PAD>', '[CLS]', '[SEP]', 'O', 'I', 'B', 'X')
# 把标签对应成数字
tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}
# 这里设置了输入到bert的最大长度
MAX_LEN = 256 - 2


# 这里是对输入的字符和标签处理，字符放在一个list，标签放在一个list
class NerDataset(Dataset):
    def __init__(self, f_path):
        with open(f_path, 'r', encoding='utf-8') as fr:
            entries = fr.read().strip().split('\n\n')
        sents, tags_li = [], [] # list of lists
        for entry in entries:
            words = [line.split()[0] for line in entry.splitlines()]
            tags = ([line.split()[-1] for line in entry.splitlines()])
            if len(words) > MAX_LEN:
                # 先对句号分段
                word, tag = [], []
                for char, t in zip(words, tags):
                    # 这里会对句号进行断开，但是之前处理数据格式的时候已经分开了每个句子，其实可以不需要
                    # 也可以思考和实验一下，用’。‘或者用’，‘哪一种断句方法会更好。
                    if char != '。':
                        if char != '\ue236':   # 测试集中有这个字符
                            word.append(char)
                            tag.append(t)
                    else:
                        sents.append(["[CLS]"] + word[:MAX_LEN] + ["[SEP]"])
                        tags_li.append(['[CLS]'] + tag[:MAX_LEN] + ['[SEP]'])
                        word, tag = [], []
                        # 最后的末尾
                if len(word):
                    sents.append(["[CLS]"] + word[:MAX_LEN] + ["[SEP]"])
                    tags_li.append(['[CLS]'] + tag[:MAX_LEN] + ['[SEP]'])
                    word, tag = [], []
            else:
                sents.append(["[CLS]"] + words[:MAX_LEN] + ["[SEP]"])
                tags_li.append(['[CLS]'] + tags[:MAX_LEN] + ['[SEP]'])
        # 这里sent存放了每一个句子字符组成的lsit，tags存放了每一个句子标签组成的list，一一对应
        self.sents, self.tags_li = sents, tags_li


    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx]
        x, y = [], []
        is_heads = []
        for w, t in zip(words, tags):
            tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = tokenizer.convert_tokens_to_ids(tokens)
            # assert len(tokens) == len(xx), f"len(tokens)={len(tokens)}, len(xx)={len(xx)}"

            # 中文没有英文wordpiece后分成几块的情况
            is_head = [1] + [0 ] *(len(tokens) - 1)
            t = [t] + ['<PAD>'] * (len(tokens) - 1)
            yy = [tag2idx[each] for each in t]  # (T,)

            x.extend(xx)
            is_heads.extend(is_head)
            y.extend(yy)
        assert len(x) == len(y) == len(is_heads), f"len(x)={len(x)}, len(y)={len(y)}, len(is_heads)={len(is_heads)}"

        # seqlen
        seqlen = len(y)

        # to string
        words = " ".join(words)
        tags = " ".join(tags)
        return words, x, is_heads, tags, y, seqlen  # 句子列表，bert的字的idx，这个字是否是开头，标签列表，数字标签列表，句子长度


    def __len__(self):
        return len(self.sents)


    # 填充
    def pad(batch):
        '''Pads to the longest sample'''
        f = lambda x: [sample[x] for sample in batch]
        words = f(0)
        is_heads = f(2)
        tags = f(3)
        seqlens = f(-1)
        maxlen = np.array(seqlens).max()

        f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
        x = f(1, maxlen)
        y = f(-2, maxlen)


        f = torch.LongTensor

        return words, f(x), is_heads, tags, f(y), seqlens
