# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 11:06:56 2022

@author: 24965
"""

import torch
import torch.nn as nn

from torch.nn import functional as F

from transformers import BertModel, BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np

class Config(object):
    
    def __init__(self, dataset):
        self.train_path = '/Users/admin/Desktop/复旦大学文本分类语料库/test/test_new.txt'
        self.test_path = '/Users/admin/Desktop/复旦大学文本分类语料库/test/test_new.txt'
        self.dev_path = '//data//dev.txt'

        self.bert_path = '/Users/admin/Desktop/bert-base-chinese-model'

        # 配置使用检测GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = 20
        
        self.num_epoch = 50
        
        # batch大小
        self.batch_size = 16
        
        #每个序列最大token数
        self.pad_size=160
        #学习率
        self.learning_rate = 2e-5
        # bert 分词器
        self.tokenizer=BertTokenizer.from_pretrained(self.bert_path) #定义分词器
        self.hidden_size=768  # Bert模型 token的embedding维度 = Bert模型后接自定义分类器（单隐层全连接网络）的输入维度
        # 每个n-gram的卷积核数量
        self.num_filters=256
        # 卷积核在序列维度上的尺寸 = n-gram大小 卷积核总数量=filter_size*num_filters
        self.filter_size=(2,3,4)
        self.dropout=0.1

class Model(nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        
        self.bert = BertModel.from_pretrained(config.bert_path)
        
        for param in self.bert.parameters():
            param.requires_grad = True # 使参数可更新
        self.convs=nn.ModuleList(
            # 输入通道数, 输出通道数（卷积核数），卷积核维度
            [nn.Conv2d(1,config.num_filters,(k,config.hidden_size)) for k in config.filter_size]    #(k,config.hidden_size)  n-gram,embedding维度
        )

        self.dropout=nn.Dropout(config.dropout)
        self.fc=nn.Linear(config.num_filters * len(config.filter_size), config.num_classes) #输入的最后一个维度，输出的最后一个维度 全连接层只改变数据的最后一个维度 由输入最后的一个维度转化为类别数
        
    def conv_and_pool(self,x,conv):
        x=conv(x)   #[batch_size,channel_num,pad_size,embedding_size（1）]
        x=F.relu(x)
        x=x.squeeze(3) #[batch_size,channel_num,pad_size]
        x=F.max_pool1d(x,x.size(2)) #经过卷积之后，x
        x = x.squeeze(2)  # [batch_size,channel_num]
        return x
    
    def forward(self, x):
        context=x[0] #batch_size*seq_length
        mask=x[2]   #batch_size*seq_length

        # 第一个参数 是所有输入对应的输出  第二个参数 是 cls最后接的分类层的输出
        encoder_out,pooled = self.bert(context,attention_mask=mask,return_dict=False) # output_all_encoded_layers 是否将bert中每层(12层)的都输出，false只输出最后一层【128*768】

        out = encoder_out.unsqueeze(1)  #增加一个维度，[batch_size,channel_num,pad_size,embedding_num]  ->  [batch_size,channel_num,pad_size,embedding_num]
        out = torch.cat([self.conv_and_pool(out,conv) for conv in self.convs],1)
        out=self.fc(out) # 128*10
        return out

config = Config("")

split_str = "__!__"

def encode_fn(text_list):
    all_input_ids = []
    for text in text_list:
        input_ids = config.tokenizer.encode(
            text,
            add_special_tokens=True,  # 添加special tokens， 也就是CLS和SEP
            max_length=config.pad_size,  # 设定最大文本长度
            pad_to_max_length=True, # pad到最大的长度
            return_tensors='pt'  # 返回的类型为pytorch tensor
        )
        all_input_ids.append(input_ids)
    all_input_ids = torch.cat(all_input_ids, dim=0)
    return all_input_ids

def data_process(text_values, labels, train):
    all_input_ids = encode_fn(text_values)
    
    labels = torch.tensor(labels)
    all_input_ids = all_input_ids.to(config.device)
    labels = labels.to(config.device)
    dataset = TensorDataset(all_input_ids, labels)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=train)
    return dataloader

def load_data(data_path):
    datas = []
    labels = []
    with open(data_path, "r",  encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            temp = line.split(split_str)
            if len(temp) == 2:
                datas.append(temp[0])
                labels.append(int(temp[1]))
    return datas, labels

'''
训练模型
'''
def train_model():
    '''
    加载新的0.8占比的训练数据，训练模型
    '''
    datas_train, labels_train = load_data(config.train_path)
    datas_train = datas_train[0:8]
    labels_train = labels_train[0:8]
    train_dataloader = data_process(datas_train, labels_train, True)
    model = Model()
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * config.num_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    model.to(config.device)
    criterion = nn.CrossEntropyLoss()

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
            outputs = model((x, None, attention_masks))
            loss = criterion(outputs, y)
            print(loss)
            total_loss = total_loss + loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        print("epoch: ", epoch, "loss: ", total_loss)
        torch.save(model.state_dict(), "./model/model" + str(epoch) + ".pkl")
        print('{{"metric": "loss", "value": {}, "epoch": {}}}'.format(total_loss, epoch))
def eval_model():
    model = Model()
    datas_test, labels_test = load_data(config.test_path)
    test_dataloader = data_process(datas_test, labels_test, False)
    model.load_state_dict(torch.load("./model/model7.pkl"))
    model.to(config.device)
    model.eval()
    p = []
    labels = []
    for step, (x, y) in enumerate(test_dataloader):
        with torch.no_grad():
            attention_masks = []
            for each in x:
                attention_mask = [1 if t != 0 else 0 for t in each]
                attention_masks.append(attention_mask)
            attention_masks = torch.from_numpy(np.array(attention_masks))
            attention_masks = attention_masks.to(config.device)
            outputs = model((x, None, attention_masks))
            outputs = outputs.to('cpu')
            label_ids = y.to('cpu').numpy()

            for j in range(len(y)):
                p.append(np.argmax(outputs[j]))
                labels.append(label_ids[j])

    print("macro precision: ", precision_score(labels, p, average='macro'))
    print("micro precision ", precision_score(labels, p, average='micro'))
    print("precision: ", precision_score(labels, p, average=None))
    print("=======================")
    print("macro f1_score: ", f1_score(labels, p, average='macro'))
    print("micro f1_score ", f1_score(labels, p, average='micro'))
    print("f1_score: ", f1_score(labels, p, average=None))
    print("=======================")
    print("macro recall_score: ", recall_score(labels, p, average='macro'))
    print("micro recall_score ", recall_score(labels, p, average='micro'))
    print("recall_score: ", recall_score(labels, p, average=None))

if __name__ == '__main__':
    train = True

    if train:
        train_model()
    else:
        eval_model()

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    