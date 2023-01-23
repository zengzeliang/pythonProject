from random import sample
import random


data_path = "../travel_data/relationship_data.txt"
relation_map = {"cr2": "人物的出生日期", "cr4": "人物的出生地", "cr16": "人物的毕业院校",
                "cr20": "人物的配偶", "cr21": "组织机构的子女", "cr28": "组织机构的高管",
                "cr29": "组织机构的员工数", "cr34": "组织机构的创始人", "cr35": "其他",
                "cr37": "组织机构的总部地点"}
def load_train_data():
    map = {}
    f = open(data_path, encoding='utf-8')
    while True:
        line = f.readline()
        line = line.strip("\n")
        if line:
            splits = line.split("\t")
            ans = map.get(splits[1], [])
            ans.append(splits[0])
            map[splits[1]] = ans
        else:
            break
    f.close()
    return map

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

def minus(list1, list2):
    # list1 - list2
    res = []
    for i in list1:
        flag = False
        for j in list2:
            if j == i:
                flag = True
                break
        if not flag:
            res.append(i)
    return res

if __name__ == '__main__':
    ans = load_train_data()
    train_data = []
    train_label = []

    test_data = []
    test_label = []

    for key in ans:
        values = ans.get(key)
        selected = sample(values, 12)
        remain = minus(values, selected)

        for i in selected:
            i = str(i)
            start1 = i.find("<e1>")
            end1 = i.find("</e1>")
            start2 = i.find("<e2>")
            end2 = i.find("</e2>")
            startEntity = i[start1 + 4: end1]
            endEntity = i[start2 + 4: end2]
            i = i.replace("<e2>", "")
            i = i.replace("</e2>", "")
            i = i.replace("<e1>", "")
            i = i.replace("</e1>", "")
            input_data = i + startEntity + "和" + endEntity + "之间的关系是[MASK]."
            train_data.append(input_data)
            train_label.append(relation_map.get(key))

        for i in remain:
            i = str(i)
            i = i.replace("<e2>", "")
            i = i.replace("</e2>", "")
            i = i.replace("<e1>", "")
            i = i.replace("</e1>", "")
            test_data.append(i + " " + key)









