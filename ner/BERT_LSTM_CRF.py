# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from my_torch_crf import CRF
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report

class Config(object):
    def __init__(self):
        self.train_path = "./cluener.train.bioes"
        self.dev_path = './cluener.dev.bioes'
        self.test_path = './cluener.test.bioes'
        self.bert_path = '/Users/admin/Desktop/bert-base-chinese-model'
        # 配置使用检测GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = len(num_labels)
        self.num_epoch = 100
        # batch大小
        self.batch_size = 16
        # 每个序列最大token数
        self.pad_size = 52
        # 学习率
        self.learning_rate = 2e-3
        # bert 分词器
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)  # 定义分词器
        self.hidden_size = 768  # Bert模型 token的embedding维度 = Bert模型后接自定义分类器（单隐层全连接网络）的输入维度
        self.lstm_hidden_size = 256
        self.dropout = 0.1
        self.train = True
        self.consecutive_no_improvement = 10

num_labels = ['E-name', 'B-game', 'B-position', 'I-book', 'E-organization', 'E-position', 'I-company', 'E-game', 'E-company', 'S-address', 'S-name', 'B-book', 'E-movie', 'I-organization', 'I-position', 'B-address', 'I-movie', 'B-name', 'I-address', 'I-government', 'I-name', 'O', 'E-address', 'B-company', 'I-scene', 'B-government', 'I-game', 'E-scene', 'E-book', 'S-company', 'S-position', 'B-scene', 'B-movie', 'B-organization', 'E-government']
num_labels_2_id = {k: v for v, k in enumerate(num_labels)}
num_id_2_labels = {v: k for v, k in enumerate(num_labels)}
config = Config()

class BERT_BiLSTM_CRF(nn.Module):
    def __init__(self):
        super(BERT_BiLSTM_CRF, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.lstm = nn.LSTM(input_size=config.hidden_size,
                            bidirectional=True,
                            hidden_size=config.lstm_hidden_size,
                            batch_first=True)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(in_features=config.lstm_hidden_size * 2, out_features=config.num_classes)
        self.crf = CRF(num_tags=config.num_classes, batch_first=True)
    def forward(self, input_ids, attention_mask, label_ids):
        embeds = self.bert(input_ids, attention_mask=attention_mask)[0]
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.fc(lstm_out)
        argmax, _ = torch.max(lstm_out, dim=2)
        loss = self.crf(lstm_out, label_ids, attention_mask.byte())
        return -loss

    def predict(self, input_ids, label_ids=None, attention_mask=None):
        embeds = self.bert(input_ids, attention_mask=attention_mask)[0]
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.fc(lstm_out)
        res = self.crf.decode(lstm_out, attention_mask.byte())
        return res
sets = set()
def load_data(path):
    sentences = []
    labels = []
    sentence = []
    label = []
    with open(path, "r",  encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            if line:
                temp = line.split(" ")
                if len(temp) == 2:
                    indexed_tokens = config.tokenizer.convert_tokens_to_ids(temp[0])
                    sentence.append(indexed_tokens)
                    label.append(num_labels_2_id[temp[1]])
                    sets.add(temp[1])
            else:
                if(len(sentence) > 0):
                    labels.append(label)
                    sentences.append(sentence)
                    sentence = []
                    label = []

    # 将句子和标签转换为 tensor
    for i in range(len(sentences)):
        if len(sentences[i]) > config.pad_size - 2:
            sentences[i] = sentences[i][:config.pad_size - 2]
            labels[i] = labels[i][:config.pad_size - 2]
        sentences[i] = [101] + sentences[i] + [102]
        labels[i] = [num_labels_2_id["O"]] + labels[i] + [num_labels_2_id["O"]]

        # 仍然小于最大数量，补充pad
        if len(sentences[i]) < config.pad_size:
            sentences[i] = sentences[i] + [0 for i in range(config.pad_size - len(sentences[i]))]

        if len(labels[i]) < config.pad_size:
            labels[i] = labels[i] + [num_labels_2_id["O"] for i in range(config.pad_size - len(labels[i]))]

        if len(sentences[i]) != len(labels[i]):
            print(sentences[i], " ", i, " ", labels[i])
    sentences = [torch.tensor(l) for l in sentences]
    labels = [torch.tensor(l) for l in labels]
    sentences = torch.stack(sentences, 0)  # 这里的维度还可以改成其它值
    labels = torch.stack(labels, 0)  # 这里的维度还可以改成其它值
    sentences = sentences.to(config.device)
    labels = labels.to(config.device)

    sentences = sentences[:64]
    labels = labels[:64]
    # 创建 TensorDataset
    dataset = TensorDataset(sentences, labels)

    shuffle = False
    if path == config.train_path:
        shuffle = True
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle)
    return dataloader

def train_model():
    train_dataloader = load_data(config.train_path)
    dev_dataloader = load_data(config.dev_path)
    model = BERT_BiLSTM_CRF()
    load_dict = torch.load("../model/mlm_pre_model.pkl", map_location=torch.device(config.device))
    new_dict = model.state_dict()
    for key, val in load_dict.items():
        new_dict[key[len("mask_"):]] = val
    model.load_state_dict(new_dict, strict=False)

    # Unfreeze bias
    for name, param in model.bert.named_parameters():
        # if name not in ["embeddings.word_embeddings.weight", "embeddings.position_embeddings.weight", "embeddings.token_type_embeddings.weight"]:
        if "bias" not in name:
            param.requires_grad = False
    #定义优化器
    optimizer = torch.optim.AdamW(filter(lambda p : p.requires_grad, model.parameters()),lr=2e-5)

    total_steps = len(train_dataloader) * config.num_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    model.to(config.device)
    model.bert.eval()
    num_steps = 0
    best_val_f1 = 0
    for epoch in range(config.num_epoch):
        model.train()
        total_loss = 0
        for step, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            attention_masks = []
            for each in x:
                attention_mask = [1 if t != 0 else 0 for t in each]
                attention_masks.append(attention_mask)
            attention_masks = torch.from_numpy(np.array(attention_masks))
            attention_masks = attention_masks.to(config.device)
            loss = model(x, attention_masks, y)
            total_loss = total_loss + loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        print(str(epoch) + ": " + "total_loss", total_loss)
        # 验证模型
        model.eval()
        real_labels = []
        predict_labels = []
        for step, (x, y) in enumerate(dev_dataloader):
            with torch.no_grad():
                attention_masks = []
                for each in x:
                    attention_mask = [1 if t != 0 else 0 for t in each]
                    attention_masks.append(attention_mask)
                attention_masks = torch.from_numpy(np.array(attention_masks))
                attention_masks = attention_masks.to(config.device)
                res = model.predict(x, y, attention_masks)
                for row_index, row in enumerate(res):
                    for j, real_label in enumerate(y[row_index]):
                        min_mask = 0
                        # attention_masks[row_index] = attention_masks[row_index].numpy()
                        for t_i, temp in enumerate(attention_masks[row_index]):
                            if temp == 0:
                                min_mask = t_i
                                break
                        if j > min_mask:
                            break
                        if real_label != num_labels_2_id["O"]:
                            real_labels.append(real_label)
                            predict_labels.append(row[j])

        real_labels = [i.item() for i in real_labels]
        print(real_labels)
        print(predict_labels)
        val_f1 = f1_score(real_labels, predict_labels, average='macro')
        r_score = recall_score(real_labels, predict_labels, average='macro')
        p_score = precision_score(real_labels, predict_labels, average='macro')

        print("p_score:", p_score)
        print("r_score:", r_score)
        print("f1:", val_f1)
        print("==========================")
        print(classification_report(real_labels, predict_labels, target_names=num_labels))
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            num_steps = 0
        else:
            num_steps = num_steps + 1
            if num_steps >= config.consecutive_no_improvement:
                break

if __name__ == '__main__':
    if config.train:
        train_model()

