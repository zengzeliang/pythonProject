from hanlp_restful import HanLPClient
hanLP = HanLPClient('https://www.hanlp.com/api', auth=None, language='zh') # auth不填则匿名，zh中文，mul多语种

if __name__ == '__main__':
    parse = hanLP.parse("杭州西湖几点开放啊")

    print(parse)

