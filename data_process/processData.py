import xlrd
import re
import math
import opencc
# 初始化简繁转换器
converter = opencc.OpenCC('t2s')
def delete_boring_characters(sentence):
    text = re.sub('\[\d+\]', " ", sentence)
    return text  # 正则匹配，将表情符合替换为空''

def tf(word, count):
    return count[word] / sum(count.values())

def idf(word, count_list):
    n_contain = sum([1 for count in count_list if word in count])
    return math.log(len(count_list) / (1 + n_contain))

def tf_idf(word, count, count_list):
    return tf(word, count) * idf(word, count_list)

def handle(segments):
    t = []
    for segment in segments:
        segment = str(segment)
        index = segment.find("/")
        t.append(segment[:index])
    return t

if __name__ == '__main__':
    path = "../travel_data/zhejiang_data_new.xlsx"
    path2 = "../travel_data/baidu_baike.xlsx"
    # 整型数字：目标sheet所在位置，以0开始，比如sheet_name = 0代表第1个工作表
    workbook = xlrd.open_workbook(path)
    sheet = workbook.sheet_by_index(0)
    documents = []

    cnt = 0
    for i in range(1, sheet.nrows):
        values = sheet.row_values(i)
        clean_str1 = delete_boring_characters(values[6])
        clean_str1 = converter.convert(clean_str1)
        one_document_sentences = clean_str1.split("。")
        document_sentences = []
        for sentence in one_document_sentences:
            if len(sentence) > 0:
                document_sentences.append(sentence)
        documents.append(document_sentences)

    workbook2 = xlrd.open_workbook(path2)
    sheet2 = workbook2.sheet_by_index(0)
    for i in range(1, sheet2.nrows):
        values = sheet.row_values(i)
        clean_str1 = delete_boring_characters(values[4])
        clean_str1 = converter.convert(clean_str1)
        one_document_sentences = clean_str1.split("。")
        document_sentences = []
        for sentence in one_document_sentences:
            if len(sentence) > 0:
                document_sentences.append(sentence)
        documents.append(document_sentences)
    # print(max_len) # 300
    #print(cnt) # 179个句子长度大于126
    # f = open("../travel_data/intro1.txt", "w")
    # f2 = open("../travel_data/intro2.txt", "w")
    # start2 = False
    # for document in documents:
    #     for sentence in document:
    #         if "湖州菰城景区以莲花庄" in sentence:
    #             start2 = True
    #         if not start2:
    #             f.write(sentence.strip() + "\n")
    #         else:
    #             f2.write(sentence.strip() + "\n")
    #     if not start2:
    #         f.write("\n")
    #     else:
    #         f2.write("\n")
    # f = open("../travel_data/intro.txt", "w")
    # for document in documents:
    #     for sentence in document:
    #         f.write(sentence.strip() + "\n")
    #     f.write("\n")