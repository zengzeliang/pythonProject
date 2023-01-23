import xlrd
import re
from pyhanlp import HanLP
from collections import Counter
import math

def delete_boring_characters(sentence):
    text = re.sub('[0-9’!"#$%&\'(（）)*+,-./:;<=>?@，。?★▲⏰、…【】《》？“”‘’！[\\]^_`{|}~\s]+', " ", sentence)
    p = re.compile(u'['u'\U0001F300-\U0001F64F' u'\U0001F680-\U0001F6FF' u'\u2600-\u2B55 \U00010000-\U0010ffff]+')
    return re.sub(p, ' ', text)  # 正则匹配，将表情符合替换为空''

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
    path = "travel_data/mafengwo_question_data.xlsx"
    path2 = "travel_data/tuniu_question_data.xlsx"
    # 整型数字：目标sheet所在位置，以0开始，比如sheet_name = 0代表第1个工作表
    workbook = xlrd.open_workbook(path)
    sheet = workbook.sheet_by_index(0)
    words = []
    for i in range(1, sheet.nrows):
        one_data = []
        values = sheet.row_values(i)
        clean_str1 = delete_boring_characters(values[0])
        # 处理分词
        segments1 = HanLP.segment(clean_str1)
        one_data.extend(handle(segments1))

        clean_str2 = delete_boring_characters(values[1])
        segments2 = HanLP.segment(clean_str2)
        one_data.extend(handle(segments2))

        clean_str3 = delete_boring_characters(values[2])
        segments3 = HanLP.segment(clean_str3)
        one_data.extend(handle(segments3))

        clean_str4 = delete_boring_characters(values[3])
        segments4 = HanLP.segment(clean_str4)
        one_data.extend(handle(segments4))
        words.append(one_data)

    workbook2 = xlrd.open_workbook(path2)
    sheet2 = workbook2.sheet_by_index(0)
    for i in range(1, sheet2.nrows):
        one_data = []
        values = sheet2.row_values(i)
        clean_str1 = delete_boring_characters(values[0])
        clean_str1.replace("\n", "")
        clean_str1.replace(" ", "")
        # 处理分词
        segments1 = HanLP.segment(clean_str1)
        one_data.extend(handle(segments1))

        clean_str2 = delete_boring_characters(values[1])
        clean_str2.replace("\n", "")
        clean_str2.replace(" ", "")
        segments2 = HanLP.segment(clean_str2)
        one_data.extend(handle(segments2))

        words.append(one_data)

    count_list = []
    for i in range(len(words)):
        count = Counter(words[i])
        count_list.append(count)
    sources = {}
    for i, count in enumerate(count_list):
        source = {word: tf_idf(word, count, count_list) for word in count}
        for k in source:
            sources[k] = float(sources.get(k, 0)) + float(source[k])

    sources = sorted(sources.items(), key=lambda x: x[1], reverse=True)
    print(sources)
