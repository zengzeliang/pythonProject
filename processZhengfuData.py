import xlrd
import xlwt

if __name__ == '__main__':
    path = "travel_data/zhejiang_data.xlsx"
    # 整型数字：目标sheet所在位置，以0开始，比如sheet_name = 0代表第1个工作表
    workbook = xlrd.open_workbook(path)
    sheet = workbook.sheet_by_index(0)
    words = []

    title_set = set()
    new_data = []
    pre_len = 0
    for i in range(sheet.nrows):
        values = sheet.row_values(i)
        title = values[3]
        title_set.add(title)
        cur_len = len(title_set)

        if cur_len > pre_len:
            new_data.append(values)
            pre_len = cur_len

    print(len(new_data))

    # 设置Excel编码
    file = xlwt.Workbook('encoding = utf-8')
    # 创建sheet工作表
    sheet1 = file.add_sheet('sheet1', cell_overwrite_ok=True)

    # 先填标题
    # sheet1.write(a,b,c) 函数中参数a、b、c分别对应行数、列数、单元格内容
    for i in range(len(new_data[0])):
        sheet1.write(0, i, new_data[0][i])  # 第1行第i列

    # 保存Excel到.py源文件同级目录
    # 循环填入数据
    for i in range(1, len(new_data)):
        for j in range(len(new_data[i])):
            sheet1.write(i, j, new_data[i][j])  # 第1列序号
    # 保存Excel到.py源文件同级目录
    file.save('travel_data/zhejiang_data_new.xlsx')
