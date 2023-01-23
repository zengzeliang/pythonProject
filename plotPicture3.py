import matplotlib.pyplot as plt

if __name__ == '__main__':

    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    # 准备数据
    x_data = ['nGQL查询返回', 'es相似问匹配', '无答案返回']
    y_data = [0.292, 0.296, 0.297]

    # 正确显示中文和负号
    plt.rcParams["axes.unicode_minus"] = False
    plt.ylim((0.25, 0.3))
    # 画图，plt.bar()可以画柱状图
    for i in range(len(x_data)):
        plt.bar(x_data[i], y_data[i], width=0.3)
    # 设置x轴标签名
    plt.xlabel("响应类型")
    # 设置y轴标签名
    plt.ylabel("平均响应时间/s")
    # 显示
    plt.show()
