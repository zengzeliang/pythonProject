import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import jieba
import re
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

dtype = torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_path = '/Users/admin/Downloads/train_new.txt'
test_path = '/Users/admin/Downloads/test_new.txt'
vocabe_path = './model/vocab.txt'

# 3 words sentences (=sequence_length is 3)

# 去除标点符号
def remove_punctuation(line, strip_all=False):
    if strip_all:
        rule = re.compile(u"[^a-zA-Z0-9\u4e00-\u9fa5]")
        line = rule.sub('', line)
    else:
        punctuation = """！？｡＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏"""
        re_punctuation = "[{}]+".format(punctuation)
        line = re.sub(re_punctuation, "", line)
    return line.strip()


split_str = "__!__"

sets = set()


def load_data(data_path):
    datas = []
    labels = []
    count = 0
    with open(data_path, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            temp = line.split(split_str)
            count = count + 1
            if len(temp) == 2:
                process_text = remove_punctuation(temp[0])
                cut_list = list(jieba.cut(process_text))
                sets.update(cut_list)
                datas.append(cut_list)
                labels.append(int(temp[1]))

            if count == 24:
                break
    return datas, labels

def loadword2Id():
    word2idx = {}
    f = open(vocabe_path, encoding='utf-8')
    while True:
        line = f.readline()
        if line.strip("\n"):
            splits = line.split(":")
            if len(splits) == 3:
                word2idx[line[:line.index(splits[2]) - 1]] = int(splits[2])
            else:
                word2idx[splits[0]] = int(splits[1])
        else:
            break
    f.close()
    return word2idx

word2idx = loadword2Id()
vocab_size = len(word2idx)

# TextCNN Parameter
embedding_size = 20
sequence_length = 160  # every sentences contains sequence_length(=3) words
num_classes = 20  # 标签类别数
batch_size = 16
filter_size = (2, 3, 4)
num_filters = 3

class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.W = nn.Embedding(vocab_size, embedding_size)

        self.convs = nn.ModuleList(
            # 输入通道数, 输出通道数（卷积核数），卷积核维度
            [nn.Conv2d(1, num_filters, (k, embedding_size)) for k in filter_size]
            # (k,config.hidden_size)  n-gram,embedding维度
        )
        # self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_size),
                            num_classes)  # 输入的最后一个维度，输出的最后一个维度 全连接层只改变数据的最后一个维度 由输入最后的一个维度转化为类别数

    def conv_and_pool(self,x,conv):
        x = conv(x)   #[batch_size,channel_num,pad_size,embedding_size（1）]
        x = F.relu(x)
        x = x.squeeze(3) #[batch_size,channel_num,pad_size]
        x = F.max_pool1d(x,x.size(2)) #经过卷积之后，x
        x = x.squeeze(2)  # [batch_size,channel_num]
        return x

    def forward(self, X):
        '''
        X: [batch_size, sequence_length]
        '''
        embedding_X = self.W(X)  # [batch_size, sequence_length, embedding_size]
        embedding_X = embedding_X.unsqueeze(1)  # add channel(=1) [batch, channel(=1), sequence_length, embedding_size]
        out = torch.cat([self.conv_and_pool(embedding_X, conv) for conv in self.convs], 1)
        out = self.fc(out)  # bash_size * num_classes
        return out

model = TextCNN().to(device)

def encode_fn(text_list):
    all_input_ids = []
    for text in text_list:
        input_ids = [word2idx[n] for n in text]
        input_ids = input_ids[0: sequence_length]
        # 小于 sequence_length 补齐
        length = len(input_ids)
        if length < sequence_length:
            input_ids.extend(word2idx.get(":") for _ in range(sequence_length - length))
        all_input_ids.append(input_ids)
    return all_input_ids

def data_process(text_values, labels, train):
    all_input_ids = encode_fn(text_values)
    labels = torch.tensor(labels)
    all_input_ids = torch.tensor(all_input_ids)
    all_input_ids = all_input_ids.to(device)
    labels = labels.to(device)
    dataset = TensorDataset(all_input_ids, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return dataloader

def train_model():
    train_data, train_labels = load_data(train_path)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    train_dataloader = data_process(train_data, train_labels, True)
    # Training
    for epoch in range(2000):
        total_loss = 0
        for batch_x, batch_y in train_dataloader:
            optimizer.zero_grad()
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            total_loss += loss
            loss.backward()
            optimizer.step()
        print('{{"metric": "loss", "value": {}, "epoch": {}}}'.format(total_loss, epoch))
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), "./model/model" + str(epoch) + ".pkl")

def eval_model():
    # Test
    model = TextCNN()
    model.load_state_dict(torch.load("./result/cnn-model/model1999.pkl"))
    test_data, test_lables = load_data(test_path)
    model.to(device)
    p = []
    labels = []
    # Predict
    model.eval()
    datas_test, labels_test = load_data(test_path)
    test_dataloader = data_process(datas_test, labels_test, False)
    for batch_x, batch_y in test_dataloader:
        with torch.no_grad():
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            outputs = outputs.to('cpu')
            label_ids = batch_y.to('cpu').numpy()

            for j in range(len(batch_y)):
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

def write_vocab_to_text():
    filename = open('./model/vocab.txt','w')#dict转txt
    for k,v in word2idx.items():
        filename.write(k+':'+str(v))
        filename.write('\n')
    filename.close()

if __name__ == '__main__':
    train = True
    if train:
        # write_vocab_to_text()
        train_model()
    else:
        eval_model()

