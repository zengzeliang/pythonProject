# -*- coding: utf-8 -*-
import random
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, AdamW, \
    get_linear_schedule_with_warmup, BertForMaskedLM, BertConfig
import numpy as np
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

sentence_split = "__!__"

# def genPretrainSentenceData():
#     path = "../travel_data/intro.txt"
#     documents = []
#     with open(path, "r") as f:
#         document = []
#         for line in f.readlines():
#             line = line.strip('\n')  # 去掉列表中每一个元素的换行符
#             if line:
#                 if len(line) < 100:
#                     document.append(line)
#                 else:
#                     line = line.replace("；", "，")
#                     line = line.replace(",", "，")
#                     line = line.replace(";", "，")
#                     line_split = line.split("，")
#                     sentence = ""
#                     for s in line_split:
#                         if len(s) < 130:
#                             sentence += s + "，"
#                         else:
#                             document.append(line[:-1] + "。")
#                             sentence = ""
#             else:
#                 if len(document) > 0:
#                     documents.append(document)
#                 document = []
#     # for doc in documents:
#     #     for j in doc:
#     #         max_len = max(max_len, len(j))
#     # print(max_len) # 99
#     random.shuffle(documents)
#     split_index = int(len(documents) * 0.8)
#     writeData(documents[:split_index], "train")
#     writeData(documents[split_index:], "dev")
#
# def writeData(documents, fileName):
#     all_text1_3_label = []
#     all_text2_3_label = []
#     all_label_3_label = []
#     all_text1_2_label = []
#     all_text2_2_label = []
#     all_label_2_label = []
#     for idx, document in enumerate(documents):
#         d_id = [i for i in range(len(documents)) if i != idx]
#         if len(document) <= 2:
#             continue
#         for sIdx, s in enumerate(document):
#             if sIdx == len(document) - 1:
#                 break
#             all_text1_2_label.append(s)
#
#             near = document[sIdx + 1]
#             all_text2_2_label.append(near)
#             all_label_2_label.append(1)
#             all_text1_3_label.append(s)
#             all_text2_3_label.append(near)
#             all_label_3_label.append(2)
#             s_id = [i for i in range(len(document)) if i != sIdx and i != sIdx + 1]
#             no_near_index = random.choice(s_id)
#             no_near_doc_index = random.choice(d_id)
#             no_near_doc_s_index = random.choice([i for i in range(len(documents[no_near_doc_index]))])
#             all_text1_2_label.append(s)
#             all_text2_2_label.append(document[no_near_index])
#             all_label_2_label.append(0)
#             all_text1_3_label.append(s)
#             all_text2_3_label.append(document[no_near_index])
#             all_label_3_label.append(1)
#             all_text1_3_label.append(s)
#             all_text2_3_label.append(documents[no_near_doc_index][no_near_doc_s_index])
#             all_label_3_label.append(0)
#
#     f = open("../travel_data/sentences2" + fileName + ".txt", "w")
#     f1 = open("../travel_data/sentences3" + fileName + ".txt", "w")
#
#     for idx in range(len(all_label_2_label)):
#         f.write(all_text1_2_label[idx] + sentence_split + all_text2_2_label[idx] + sentence_split + str(all_label_2_label[idx]) + "\n")
#
#     for idx in range(len(all_label_3_label)):
#         f1.write(all_text1_3_label[idx] + sentence_split + all_text2_3_label[idx] + sentence_split + str(all_label_3_label[idx]) + "\n")

class Config(object):
    def __init__(self):
        self.sentence2_data_path = "../travel_data/sentences2train.txt"
        self.sentence2_data_dev_path = "../travel_data/sentences2dev.txt"
        self.sentence3_data_path = "../travel_data/sentences3train.txt"
        self.sentence3_data_dev_path = "../travel_data/sentences3dev.txt"
        self.bert_path = '/Users/admin/Desktop/bert-base-chinese-model'
        self.pretrain_mlm_model = '../model/mlm_pre_model_rel.pkl'
        self.sentence_pair = 2
        self.model_save_path = "../model/mlm_nsp" + str(self.sentence_pair) + "_pre_model.pkl"
        # 配置使用检测GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epoch = 1

        # batch大小
        self.batch_size = 8
        self.pad_size = 200
        # 学习率
        self.learning_rate = 2e-3
        # bert 分词器
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)  # 定义分词器
        self.maskModel = BertForMaskedLM.from_pretrained(self.bert_path)
        self.hidden_size = 768  # Bert模型 token的embedding维度 = Bert模型后接自定义分类器（单隐层全连接网络）的输入维度
        self.lstm_hidden_size = 256
        self.dropout = 0.1
        self.train = True
        self.consecutive_no_improvement = 3
        self.max_mask = 5

my_config = Config()

def load_data(path):
    sentences = []
    labels = []
    with open(path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            if line:
                temp = line.split(sentence_split)
                if len(temp) == 3:
                    sentences.append(temp[0] + sentence_split + temp[1])
                    labels.append(int(temp[2]))
    sentences_t = []
    # 将句子和标签转换为 tensor
    for sentence in sentences:
        two_sentence = sentence.split(sentence_split)
        if len(two_sentence) == 2:
            ids0 = my_config.tokenizer.encode(two_sentence[0], add_special_tokens = False)
            ids1 = my_config.tokenizer.encode(two_sentence[1], add_special_tokens = False)
            new_sentence = [101] + ids0 + [102] + ids1 + [102]
            if len(new_sentence) < my_config.pad_size:
                new_sentence = new_sentence + [0] * (my_config.pad_size - len(new_sentence))
            sentences_t.append(new_sentence)

    sentences_t = [torch.tensor(l) for l in sentences_t]
    # 创建 TensorDataset
    sentences_t = torch.stack(sentences_t, 0)

    labels = [torch.tensor(l) for l in labels]
    labels = torch.stack(labels, 0)
    sentences_t = sentences_t[:8]
    labels = labels[:8]
    dataset = TensorDataset(sentences_t, labels)
    shuffle = False
    if path == my_config.train:
        shuffle = True
    dataloader = DataLoader(dataset, batch_size=my_config.batch_size, shuffle=shuffle)
    return dataloader

class BertForNextSentencePrediction(nn.Module):
    def __init__(self, config):
        super(BertForNextSentencePrediction, self).__init__()
        self.bert = BertModel.from_pretrained(my_config.bert_path)
        self.classifier = nn.Linear(config.hidden_size, my_config.sentence_pair)
        self.config = config
    def forward(self,
                input_ids,  # [src_len, batch_size]
                attention_mask=None,  # [batch_size, src_len] mask掉padding部分的内容
                token_type_ids=None,  # [src_len, batch_size] 如果输入模型的只有一个序列，那么这个参数也不用传值
                position_ids=None,
                next_sentence_labels=None):  # [batch_size,]
        pooled_output, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, return_dict=False)
        cls_token = pooled_output[:, 0, :]
        seq_relationship_score = self.classifier(cls_token)
        # seq_relationship_score: [batch_size, 3]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(seq_relationship_score.view(-1, my_config.sentence_pair), next_sentence_labels.view(-1))
        return loss

    def predict(self,
                input_ids,  # [src_len, batch_size]
                attention_mask=None,  # [batch_size, src_len] mask掉padding部分的内容
                token_type_ids=None,  # [src_len, batch_size] 如果输入模型的只有一个序列，那么这个参数也不用传值
                position_ids=None,
                next_sentence_labels=None):  # [batch_size,]
        pooled_output, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                     position_ids=position_ids, return_dict=False)
        cls_token = pooled_output[:, 0, :]
        seq_relationship_score = self.classifier(cls_token)

        predict_view = torch.argmax(seq_relationship_score, dim=1)
        predict_view = [l.item() for l in predict_view]
        labels_view = next_sentence_labels.view(-1)
        labels_view = [l.item() for l in labels_view]
        return predict_view, labels_view

class BertForLMTransformHead(nn.Module):
    def __init__(self, config):
        super(BertForLMTransformHead, self).__init__()
        self.config = config
        self.mask_bert = BertForMaskedLM.from_pretrained(my_config.bert_path)
        self.criterion = nn.NLLLoss(ignore_index=0) # 忽略标签0的影响

    def forward(self, input_ids = None, attention_mask = None, token_type_ids = None, labels = None, s_mask_indexs = None):
        outputs = self.mask_bert(input_ids, attention_mask, token_type_ids).logits
        s_mask_indexs = torch.tensor(s_mask_indexs)
        s_mask_indexs = s_mask_indexs.to(my_config.device)
        outputs_new = []
        labels_new = []
        for idx, batch in enumerate(outputs):
            selected = batch.index_select(dim=0, index=s_mask_indexs[idx])
            outputs_new.append(selected)
            labels_new.append(labels[idx].index_select(dim=0, index=s_mask_indexs[idx]))
        outputs_new = torch.cat(outputs_new, dim=0)
        labels_new = torch.cat(labels_new, dim=0)
        log_probs = nn.LogSoftmax(dim=1)(outputs_new)
        # 只选取mask的位置
        loss = self.criterion(log_probs, labels_new)
        return loss

    def predict(self, input_ids = None, attention_mask = None, token_type_ids = None, labels = None, s_mask_indexs = None):
        outputs = self.mask_bert(input_ids, attention_mask, token_type_ids).logits
        # 只选取mask的位置
        s_mask_indexs = torch.tensor(s_mask_indexs)
        s_mask_indexs = s_mask_indexs.to(my_config.device)
        outputs_new = []
        labels_new = []
        for idx, batch in enumerate(outputs):
            selected = batch.index_select(dim=0, index=s_mask_indexs[idx])
            outputs_new.append(selected)
            labels_new.append(labels[idx].index_select(dim=0, index=s_mask_indexs[idx]))
        predicted = []
        outputs_new = torch.cat(outputs_new, dim=0)
        labels_new = torch.cat(labels_new, dim=0)
        for batch in outputs_new:
            p = torch.argmax(batch).item()
            predicted.append(p)

        return predicted, labels_new

def train_model():
    path1 = ""
    path2 = ""
    if my_config.sentence_pair == 2:
        path1 = my_config.sentence2_data_path
        path2 = my_config.sentence2_data_dev_path
    elif my_config.sentence_pair == 3:
        path1 = my_config.sentence3_data_path
        path2 = my_config.sentence3_data_dev_path
    train_dataloader = load_data(path1)
    val_dataloader = load_data(path2)
    config = BertConfig.from_pretrained(my_config.bert_path)
    mlmModel = BertForLMTransformHead(config)

    optimizer = AdamW(mlmModel.parameters(), lr=my_config.learning_rate)
    total_steps = len(train_dataloader) * my_config.num_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    mlmModel.to(my_config.device)
    best_val_f1 = 0
    for epoch in range(my_config.num_epoch):
        mlmModel.train()
        for step, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            attention_masks = []
            segment_ids = []
            origins = []
            s_mask_indexs = []
            for each in x:
                attention_mask = [1 if t != 0 else 0 for t in each]
                attention_masks.append(attention_mask)
                segment_id = []
                origin = []
                startS2 = False
                for z in each:
                    if z == 102:
                        startS2 = True
                    if not startS2:
                        segment_id.append(0)
                    else:
                        segment_id.append(1)
                # 未被padding，并且不是，特殊字符[CLS][SEP]
                waiting_choice_index = []
                for idx, t in enumerate(each):
                    if t != 0 and t != 101 and t != 102:
                        waiting_choice_index.append(idx)
                n_mask = min(my_config.max_mask, max(1, int(len(waiting_choice_index) * 0.15)))
                random.shuffle(waiting_choice_index)
                mask_indexs = waiting_choice_index[:n_mask]
                if len(mask_indexs) < my_config.max_mask:
                    mask_indexs = mask_indexs + [0] * (my_config.max_mask - len(mask_indexs))
                s_mask_indexs.append(mask_indexs)
                for idx, t in enumerate(each):
                    origin.append(t.clone())
                    if idx in mask_indexs:
                        if random.random() < 0.8:
                            each[idx] = 103
                        elif random.random() > 0.9:
                            index = random.randint(0, config.vocab_size - 1)
                            while index == 0 or index == 101 or index == 102 or index == 103:
                                index = random.randint(0, config.vocab_size - 1)
                            each[idx] = index
                segment_ids.append(segment_id)
                origins.append(origin)
            attention_masks = torch.from_numpy(np.array(attention_masks))
            attention_masks = attention_masks.to(my_config.device)
            segment_ids = torch.from_numpy(np.array(segment_ids))
            segment_ids = segment_ids.to(my_config.device)
            origins = torch.from_numpy(np.array(origins))
            origins = origins.to(my_config.device)
            x = x.to(my_config.device)
            loss = mlmModel(x, attention_masks, segment_ids, origins, s_mask_indexs)
            print('{{"metric": "loss", "value": {}, "epoch": {}}}'.format(loss, epoch))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlmModel.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # 验证模型
        mlmModel.eval()
        real_labels = []
        predict_labels = []
        num_steps = 0
        for step, (x, y) in enumerate(val_dataloader):
            with torch.no_grad():
                attention_masks = []
                segment_ids = []
                origins = []
                s_mask_indexs = []
                for each in x:
                    attention_mask = [1 if t != 0 else 0 for t in each]
                    attention_masks.append(attention_mask)
                    segment_id = []
                    origin = []
                    startS2 = False
                    for z in each:
                        if z == 102:
                            startS2 = True
                        if not startS2:
                            segment_id.append(0)
                        else:
                            segment_id.append(1)
                    # 未被padding，并且不是，特殊字符[CLS][SEP]
                    waiting_choice_index = []
                    for idx, t in enumerate(each):
                        if t != 0 and t != 101 and t != 102:
                            waiting_choice_index.append(idx)
                    n_mask = min(my_config.max_mask, max(1, int(len(waiting_choice_index) * 0.15)))
                    random.shuffle(waiting_choice_index)
                    mask_indexs = waiting_choice_index[:n_mask]
                    if len(mask_indexs) < my_config.max_mask:
                        mask_indexs = mask_indexs + [0] * (my_config.max_mask - len(mask_indexs))
                    s_mask_indexs.append(mask_indexs)
                    for idx, t in enumerate(each):
                        origin.append(t.clone())
                        if idx in mask_indexs:
                            if random.random() < 0.8:
                                each[idx] = 103
                            elif random.random() > 0.9:
                                index = random.randint(0, config.vocab_size - 1)
                                while index == 0 or index == 101 or index == 102 or index == 103:
                                    index = random.randint(0, config.vocab_size - 1)
                                each[idx] = index

                    segment_ids.append(segment_id)
                    origins.append(origin)
                attention_masks = torch.from_numpy(np.array(attention_masks))
                attention_masks = attention_masks.to(my_config.device)
                segment_ids = torch.from_numpy(np.array(segment_ids))
                segment_ids = segment_ids.to(my_config.device)
                origins = torch.from_numpy(np.array(origins))
                origins = origins.to(my_config.device)
                x = x.to(my_config.device)
                p, tr = mlmModel.predict(x, attention_masks, segment_ids, origins, s_mask_indexs)
                real_labels.extend(tr)
                predict_labels.extend(p)
        real_labels = [i.item() for i in real_labels]
        val_f1 = f1_score(real_labels, predict_labels, average='macro')
        r_score = recall_score(real_labels, predict_labels, average='macro')
        p_score = precision_score(real_labels, predict_labels, average='macro')

        print("p_score:", p_score)
        print("r_score:", r_score)
        print("f1:", val_f1)
        print("==========================")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            num_steps = 0
            torch.save(mlmModel.state_dict(), my_config.model_save_path)
        else:
            num_steps = num_steps + 1
            if num_steps >= my_config.consecutive_no_improvement:
                break

def train_nsp():
    train_dataloader = load_data(my_config.sentence2_data_path)
    val_dataloader = load_data(my_config.sentence2_data_dev_path)
    config = BertConfig.from_pretrained(my_config.bert_path)
    nspModel = BertForNextSentencePrediction(config)
    # 从MLM任务开始训练
    load_dict = torch.load(my_config.pretrain_mlm_model, map_location=torch.device(my_config.device))
    new_dict = nspModel.state_dict()
    for key, val in load_dict.items():
        new_dict[key[len("mask_"):]] = val
    nspModel.load_state_dict(new_dict, strict=False)

    optimizer = AdamW(nspModel.parameters(), lr=my_config.learning_rate)
    total_steps = len(train_dataloader) * my_config.num_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    nspModel.to(my_config.device)
    best_val_f1 = 0
    for epoch in range(my_config.num_epoch):
        nspModel.train()
        for step, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            attention_masks = []
            segment_ids = []
            position_ids = []
            for each in x:
                attention_mask = [1 if t != 0 else 0 for t in each]
                attention_masks.append(attention_mask)
                segment_id = []
                startS2 = False
                for z in each:
                    if z == 102:
                        startS2 = True
                    if not startS2:
                        segment_id.append(0)
                    else:
                        segment_id.append(1)
                position_ids.append([i for i in range(len(attention_mask))])
                segment_ids.append(segment_id)

            attention_masks = torch.from_numpy(np.array(attention_masks))
            attention_masks = attention_masks.to(my_config.device)
            position_ids = torch.from_numpy(np.array(position_ids))
            position_ids = position_ids.to(my_config.device)
            segment_ids = torch.from_numpy(np.array(segment_ids))
            segment_ids = segment_ids.to(my_config.device)
            x = x.to(my_config.device)
            y = y.to(my_config.device)
            loss = nspModel(x, attention_masks, segment_ids, position_ids, y)
            print('{{"metric": "loss", "value": {}, "epoch": {}}}'.format(loss, epoch))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nspModel.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # 验证模型
        nspModel.eval()
        real_labels = []
        predict_labels = []
        num_steps = 0
        for step, (x, y) in enumerate(val_dataloader):
            with torch.no_grad():
                attention_masks = []
                segment_ids = []
                position_ids = []
                for each in x:
                    attention_mask = [1 if t != 0 else 0 for t in each]
                    attention_masks.append(attention_mask)
                    segment_id = []
                    startS2 = False
                    for z in each:
                        if z == 102:
                            startS2 = True
                        if not startS2:
                            segment_id.append(0)
                        else:
                            segment_id.append(1)
                    position_ids.append([i for i in range(len(attention_mask))])
                    segment_ids.append(segment_id)

                attention_masks = torch.from_numpy(np.array(attention_masks))
                attention_masks = attention_masks.to(my_config.device)
                position_ids = torch.from_numpy(np.array(position_ids))
                position_ids = position_ids.to(my_config.device)
                segment_ids = torch.from_numpy(np.array(segment_ids))
                segment_ids = segment_ids.to(my_config.device)
                x = x.to(my_config.device)
                y = y.to(my_config.device)
                p, tr = nspModel.predict(x, attention_masks, segment_ids, position_ids, y)
                real_labels.extend(tr)
                predict_labels.extend(p)

        val_f1 = f1_score(real_labels, predict_labels, average='macro')
        r_score = recall_score(real_labels, predict_labels, average='macro')
        p_score = precision_score(real_labels, predict_labels, average='macro')

        print("p_score:", p_score)
        print("r_score:", r_score)
        print("f1:", val_f1)
        print("==========================")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            num_steps = 0
            torch.save(nspModel.state_dict(), my_config.model_save_path)
        else:
            num_steps = num_steps + 1
            if num_steps >= my_config.consecutive_no_improvement:
                break

if __name__ == '__main__':
    # genPretrainSentenceData()
    if my_config.train:
        train_nsp()