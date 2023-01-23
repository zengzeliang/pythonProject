import matplotlib.pyplot as plt
import numpy

if __name__ == '__main__':
    # 折线图
    x = [(a + 1) * 10 for a in range(5)]
    # x = [1, 7, 11, 17, 19, 25]  # 点的横坐标
    k2 = [0.1726, 0.17215352, 0.1736448335647583, 0.173832945823665, 0.1762308406829834]

    # plt.plot(x, k1, 's-', color='r', label="cpu")  # s-:方形
    # plt.plot(x, k2, 'o-', color='#2F86FF', label="loss_epoach")  # o-:圆形
    plt.ylim((0.17, 0.18))
    plt.plot(x, k2, 'o-', color='#1776B6', label="cpu")  # o-:圆形
    plt.xlabel("text_length")  # 横坐标名字
    plt.ylabel("avg_response_time/s")  # 纵坐标名字
    plt.legend(loc="best")  # 图例
    plt.savefig('./picture/intention_response_time.png')
    plt.show()

