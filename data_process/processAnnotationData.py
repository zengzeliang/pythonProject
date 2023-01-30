# -*- coding: utf-8 -*-

path = "/Users/admin/Desktop/tag_data/travel_note/intro1.ann"
def extract_data():
    documents = []
    with open(path, "r") as f:
        document = []
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            if line:
                one_line = line.split("\t")

if __name__ == '__main__':
    extract_data()
