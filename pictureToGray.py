# encoding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = '/Users/admin/Desktop/Thesis-SE-ZJU-LaTeX-master/figures/studio.png'
    # 读取原始图像
    img = cv2.imread(path)

    # 获取图像高度和宽度
    height = img.shape[0]
    width = img.shape[1]

    # 创建一幅图像
    grayimg = np.zeros((height, width, 3), np.uint8)

    # 图像平均灰度处理方法
    for i in range(height):
        for j in range(width):
            # 灰度值为RGB三个分量的平均值
            gray = (int(img[i, j][0]) + int(img[i, j][1]) + int(img[i, j][2])) / 3
            grayimg[i, j] = np.uint8(gray)

    # 显示图像
    # cv2.imshow("src", img)
    # cv2.imshow("gray", grayimg)

    cv2.imwrite('/Users/admin/Desktop/Thesis-SE-ZJU-LaTeX-master/figures/studio1.png', grayimg)

    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()



