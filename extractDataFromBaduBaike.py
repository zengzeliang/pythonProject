import xlrd
import xlwt
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import time
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager

if __name__ == '__main__':
    path = "travel_data/zhejiang_data_new.xlsx"
    # path = "/Users/admin/Desktop/景点对齐数据.xlsx"
    # 整型数字：目标sheet所在位置，以0开始，比如sheet_name = 0代表第1个工作表
    workbook = xlrd.open_workbook(path)
    sheet = workbook.sheet_by_index(0)
    browser = webdriver.Chrome(ChromeDriverManager().install())
    url = "https://baike.baidu.com"
    AllUnivInfolist = []
    count = 16
    try:
        while count < sheet.nrows:
            try:
                browser.get(url)
                while count < sheet.nrows:
                    values = sheet.row_values(count)
                    title_class = str(values[1])
                    title = str(values[3])
                    index = title.find("A")
                    if index != -1:
                        title = title[0 : index - 1]
                    if title_class != "百县千碗":
                            time.sleep(2)
                            browser.find_element(By.ID, "query").send_keys(title)  # 找到输入框输入字段
                            time.sleep(3)
                            browser.find_element(By.ID, 'search').send_keys(Keys.ENTER)  # 找到搜索按钮模拟点击
                            html = browser.page_source  # 获取html页面

                            soup = BeautifulSoup(html, 'html.parser')  # beautifulsoup库解析html

                            title = soup.find_all('dt', class_="basicInfo-item name")  # key
                            node = soup.find_all('dd', class_="basicInfo-item value")  # value

                            allunivinfo = []
                            titlelist = []
                            infolist = []

                            for i in title:  # 将所有dt标签内容存入列表
                                title = i.get_text()
                                titlelist.append(title)
                            for i in node:  # 将所有dd标签内容存入列表
                                info = i.get_text()
                                infolist.append(info)
                            for i, j in zip(titlelist, infolist):  # 多遍历循环，zip()接受一系列可迭代对象作为参数，将对象中对应的元素打包成一个个tuple（元组），然后返回由这些tuples组成的list（列表）。
                                info = ''.join((str(i) + ':' + str(j)).split())
                                allunivinfo.append(info)
                            AllUnivInfolist.append(allunivinfo)
                            # 模拟ctrl+a 操作 全选输入框内容
                            browser.find_element(By.ID, 'query')
                            actions = ActionChains(browser)
                            actions.key_down(Keys.CONTROL).send_keys('a').key_up(Keys.CONTROL)
                            actions.perform()
                            time.sleep(2)
                            # 删除输入框内容 (删除操作 模拟键盘的Backspace)
                            browser.find_element(By.ID, 'query').send_keys(Keys.BACK_SPACE)
                            count = count + 1
                            AllUnivInfolist.append(allunivinfo)
            finally:
                count = count + 1
    finally:
        browser.refresh()
        browser.quit()

        # 设置Excel编码
        file = xlwt.Workbook('encoding = utf-8')
        # 创建sheet工作表
        sheet1 = file.add_sheet('sheet1', cell_overwrite_ok=True)
        # 保存Excel到.py源文件同级目录
        # 循环填入数据
        for i in range(len(AllUnivInfolist)):
            for j in range(len(AllUnivInfolist[i])):
                sheet1.write(i, j, AllUnivInfolist[i][j])  # 第1列序号
        # 保存Excel到.py源文件同级目录
        file.save('travel_data/baike_data.xls')