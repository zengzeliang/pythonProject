import torch

from flask import request, jsonify, Flask
from transformers import BertTokenizer

from travel import Model, config, data_process

from transformers import BertModel

import numpy as np

app = Flask(__name__)

bert_path = '/Users/admin/Desktop/bert-base-chinese-model'
tokenizer = BertTokenizer.from_pretrained(bert_path) #定义分词器

bert = BertModel.from_pretrained(config.bert_path)

#解析请求参数
def request_parse(req_data):
    if req_data.method == 'POST':
        data = req_data.json
    elif req_data.method == 'GET':
        data = req_data.args
    return data

@app.route('/', methods = ["GET","POST"])   # GET 和 POST 都可以
def get_data():
    # 假设有如下 URL
    data = request_parse(request)
    # 可以通过 request 的 args 属性来获取参数
    question = data.get("question")
    test_dataloader = data_process([question], [-1], False)

    for step, (x, y) in enumerate(test_dataloader):
        with torch.no_grad():
            attention_masks = []
            for each in x:
                attention_mask = [1 if t != 0 else 0 for t in each]
                attention_masks.append(attention_mask)
            attention_masks = torch.from_numpy(np.array(attention_masks))
            attention_masks = attention_masks.to(config.device)

            # 第一个参数 是所有输入对应的输出  第二个参数 是 cls最后接的分类层的输出
            encoder_out, pooled = bert(x, attention_mask=attention_masks, return_dict=False)
            pooled = torch.squeeze(pooled, dim=0)
            return "向量为: " + str(pooled.numpy())

if __name__ == '__main__':
    app.run(port=3031, host="127.0.0.1")
