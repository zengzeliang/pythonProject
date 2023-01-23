import os
import random

import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np

label_id_to_name = {}
label_name_to_id = {}

bert_path = "/Users/admin/Desktop/bert-base-chinese-model"

split_str = "__!__"

num_labels = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def data_process(text_values, labels, train):
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    batch_size = 8
    def encode_fn(text_list):
        all_input_ids = []
        for text in text_list:
            input_ids = tokenizer.encode(
                text,
                add_special_tokens=True,  # 添加special tokens， 也就是CLS和SEP
                max_length=256,  # 设定最大文本长度
                pad_to_max_length=True, # pad到最大的长度
                return_tensors='pt'  # 返回的类型为pytorch tensor
            )
            all_input_ids.append(input_ids)
        all_input_ids = torch.cat(all_input_ids, dim=0)
        return all_input_ids

    all_input_ids = encode_fn(text_values)

    labels = torch.tensor(labels)
    all_input_ids = all_input_ids.to(device)
    labels = labels.to(device)
    dataset = TensorDataset(all_input_ids, labels)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train)

    return dataloader

'''
训练模型
'''
def train_model():
    model = BertForSequenceClassification.from_pretrained(bert_path, num_labels=num_labels, output_attentions=False,
                                                          output_hidden_states=True)
    '''
    加载新的0.8占比的训练数据，训练模型
    '''
    datas_train, labels_train = load_data('/Users/admin/Desktop/复旦大学文本分类语料库/test/test_new.txt')
    datas_train = datas_train[0:8]
    labels_train = labels_train[0:8]
    train_dataloader = data_process(datas_train, labels_train, True)
    epochs = 20
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        # 开始训练
        for step, batch in enumerate(train_dataloader):
            print(batch[0])
            # 梯度清零
            model.zero_grad()
            # 计算loss
            res = model(batch[0], token_type_ids=None, attention_mask=(batch[0] > 0),
                        labels=batch[1])
            loss = res[0].item()
            total_loss += loss
            # 梯度回传
            res[0].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # 梯度更新
            optimizer.step()
            scheduler.step()
        print("epoch: ",  epoch, "loss: ", total_loss)
        torch.save(model.state_dict(), "./model" + str(epoch) + ".pkl")

#训练数据集
def eval_model():
    model = BertForSequenceClassification.from_pretrained(bert_path, num_labels=num_labels, output_attentions=False,
                                                          output_hidden_states=True)
    datas_test, labels_test = load_data('/Users/admin/Desktop/复旦大学文本分类语料库/test/test_new.txt')
    print(len(datas_test))
    test_dataloader = data_process(datas_test, labels_test, False)
    model.load_state_dict(torch.load("./model.pkl"))
    model.to(device)
    model.eval()
    p = []
    labels = []
    for batch in test_dataloader:
        with torch.no_grad():
            res = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0] > 0),
                        labels=batch[1])

            ans = res[1].cpu().numpy()
            label_ids = batch[1].to('cpu').numpy()

            for j in range(len(ans)):
                p.append(np.argmax(ans[j]))
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

'''
随机划分数据集
'''
def data_split(full_list, ratio, shuffle=False):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     子列表1
    :param shuffle:   子列表2
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2

def readData():
    data_dir_train = '/Users/admin/Desktop/复旦大学文本分类语料库/train/'
    data_dir_test = '/Users/admin/Desktop/复旦大学文本分类语料库/test/'
    file_list = []

    # 加载训练数据
    for root, file, files in os.walk(data_dir_train):
        file_list.append(root)
    file_list = file_list[1:]

    for idx in range(len(file_list)):
        temp = file_list[idx][file_list[idx].rindex("/") + 1:]
        label_name_to_id[temp] = idx
        label_id_to_name[idx] = temp

    # 加载测试数据
    first = True
    for root, file, files in os.walk(data_dir_test):
        if first is not True:
            file_list.append(root)
        first = False

    texts_labels = []
    split_str = "__!__"
    for file in file_list:
        category = file[file.rindex("/") + 1:]

        for corpus in os.listdir(file):
            text = open(file + "/" + corpus, "rb").read().decode('GB2312', 'ignore')
            # 去除一些奇怪的字符
            text = text.replace('\r', ' ').replace('\n', ' ').replace('\u3000', '').replace('             ', '')
            texts_labels.append(text + split_str + str(label_name_to_id[category]))

    train_new, test_new = data_split(texts_labels, 0.8, True)
    print("total_datas:", len(texts_labels))
    with open(data_dir_train + "train_new.txt", 'w') as f:
        for i in train_new:
            f.write(i + '\n')

    with open(data_dir_test + "test_new.txt", 'w') as f:
        for i in test_new:
            f.write(i + '\n')

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

if __name__ == '__main__':

    '''
    1. 初始处理数据，将数据训练集和测试集数据合成统一数据，再随机划分为新的训练集和数据集8:2，保存在text文件中
    '''
    # readData()

    train = True

    if train:
        train_model()
    else:
        eval_model()









