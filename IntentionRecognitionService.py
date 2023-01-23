import time

import torch

from flask import request, Flask

from travel import Model, config, data_process
import numpy as np

app = Flask(__name__)

model = Model()
model_name = 'model9.pkl'
intention_text_path = "./question_data/intention_map.txt"
model.load_state_dict(torch.load("./travel-model/" + model_name, map_location=torch.device(config.device)))
model.to(config.device)
model.eval()

def load_intention_id_map():
    intention_id_map = {}
    f = open(intention_text_path, encoding='utf-8')
    while True:
        line = f.readline()
        if line.strip("\n"):
            splits = line.split(" ")
            intention_id_map[int(splits[0])] = splits[1]
        else:
            break
    f.close()
    return intention_id_map
intention_id_map = load_intention_id_map()

#解析请求参数
def request_parse(req_data):
    if req_data.method == 'POST':
        data = req_data.json
    elif req_data.method == 'GET':
        data = req_data.args
    return data
total = 0
pre = 0
count = 0
 # GET 和 POST 都可以
@app.route('/', methods = ["GET","POST"])
def get_data():
    start = time.time()
    data = request_parse(request)
    # 通过 request 的 args 属性来获取参数
    question = data.get("question")
    test_dataloader = data_process([question], [-1], False)
    intention_id = 0
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
            for j in range(len(y)):
                intention_id = np.argmax(outputs).item()
    end = time.time()
    global total
    global pre
    global count
    total = total + (end - start)
    count = count + 1
    if count == 10:
        print((total - pre) / 10)
        pre = total
        count = 0
    return intention_id_map.get(intention_id)

if __name__ == '__main__':
    app.run(port=3030, host="127.0.0.1")

