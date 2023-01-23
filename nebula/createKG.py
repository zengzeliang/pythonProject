# -*- coding:utf-8 -*-
import time

import xlrd
from nebula2.gclient.net import ConnectionPool
from nebula2.Config import Config
import pymysql
import requests
import re

def clearText(text):
    pattern = r'\[\d+\]'
    result = re.sub(pattern, "", text)
    return result

def parseSightLevel(sightLevel):
    sightLevel = sightLevel.upper()
    index = sightLevel.find("A")
    if index != -1:
        num = 0
        if index >= 1 and sightLevel[index - 1].isnumeric():
            num = int(sightLevel[index - 1])
        else:
            while index < len(sightLevel):
                if sightLevel[index] == 'A':
                    num = num + 1
                    index = index + 1
                else:
                    break
        res = ""
        for i in range(num):
            res = res + "A"
        return res
    else:
        return sightLevel

if __name__ == '__main__':
    # 连接mysql数据库
    db = pymysql.connect(host='localhost', user='root', password='123456', database='travel')
    cursor = db.cursor()
    # 定义配置
    config = Config()
    config.max_connection_pool_size = 10
    # 初始化连接池
    connection_pool = ConnectionPool()
    # 如果给定的服务器正常，则返回true，否则返回false。
    ok = connection_pool.init([('127.0.0.1', 9669)], config)
    # 从连接池中获取会话
    session = connection_pool.get_session('root', 'nebula')
    txt_name = "name_diff.txt"
    f = open(txt_name, "w")
    vid = 1000
    eid = -1000
    city_map = {}
    climate_map = {}
    district_map = {}
    food_map = {}
    level_map = {}
    province_map = {}
    sight_map = {}
    addr_map = {}
    name_diff = {}

    sight = 'INSERT VERTEX sight(id, name, type, addr, lng, lat) VALUE "{}":({}, "{}", "{}", "{}", "{}", "{}")'
    season = 'INSERT VERTEX season(id, attribute) VALUE "{}":({}, "{}")'
    province = 'INSERT VERTEX province(id, name) VALUE "{}":({}, "{}")'
    level = 'INSERT VERTEX level(id, attribute) VALUE "{}":({}, "{}")'
    food = 'INSERT VERTEX food(id, name) VALUE "{}":({}, "{}")'
    district = 'INSERT VERTEX district(id, name) VALUE "{}":({}, "{}")'
    climate = 'INSERT VERTEX climate(id, attribute) VALUE "{}":({}, "{}")'
    city = 'INSERT VERTEX city(id, name) VALUE "{}":({}, "{}")'
    object1 = 'INSERT VERTEX attributes(id, attribute) VALUE "{}":({}, "{}")'
    relation = 'INSERT EDGE relation(id, name) VALUES "{}" -> "{}":({}, "{}")'

    start_row = 1
    try:
        # 选择图空间
        session.execute('USE travel')
        # 执行查看TAG命令
        # result = session.execute('SHOW spaces')

        path1 = "../travel_data/zhejiang_data_new.xlsx"
        path2 = "../travel_data/baidu_baike.xlsx"
        workbook = xlrd.open_workbook(path1)
        sheet = workbook.sheet_by_index(0)

        workbook2 = xlrd.open_workbook(path2)
        sheet2 = workbook2.sheet_by_index(0)

        for i in range(start_row, sheet2.nrows):
            time.sleep(0.1)
            values = sheet.row_values(i)
            values2 = ["" for i in range(20)]
            response = requests.get("https://restapi.amap.com/v3/geocode/geo?address=浙江"+values[8]+"&key=24290640af32455b75766462614a9bdf")
            text = str(response.content, "utf-8")
            addr_dict = eval(text)
            if addr_dict["status"] != "1":
                break
            print(str(i), " ", addr_dict)
            sightVid = -1
            foodVid = -1
            locations = str(addr_dict["geocodes"][0]["location"])
            provinceName = str(addr_dict["geocodes"][0]["province"])
            if values[1] == "风景名胜" or values[1] == "红色旅游" or values[1] == "美丽乡村":
                values2 = sheet2.row_values(i)
                split = locations.split(",")
                # 不存在此景点
                if sight_map.get(values[3], -1) == -1:
                    if values2[3] != "":
                        name_diff[values[3]] = values2[3]
                    sightVid = vid
                    sight_map[values[3]] = sightVid
                    vid = vid + 1
                else:
                    sightVid = sight_map[values[3]]
                sight_format = sight.format(sightVid, sightVid, values[3], values[1], values[8], split[0], split[1])

            elif values[1] == "A级景区":
                values2 = sheet2.row_values(i)
                values_split = values[3].split("\xa0\xa0")
                if len(values_split) == 1:
                    values_split = values[3].split(" ")
                values[3] = values_split[0]
                values2[16] = values_split[1]
                split = locations.split(",")
                # 不存在此景点
                if sight_map.get(values[3], -1) == -1:
                    if values2[3] != "":
                        name_diff[values[3]] = values2[3]
                    sightVid = vid
                    sight_map[values[3]] = sightVid
                    vid = vid + 1
                else:
                    sightVid = sight_map[values[3]]
                sight_format = sight.format(sightVid, sightVid, values[3], "风景名胜", values[8], split[0], split[1])

            elif values[1] == "百县千碗":
                if food_map.get(values[3], -1) == -1:
                    foodVid = vid
                    food_map[values[3]] = foodVid
                    vid = vid + 1
                else:
                    foodVid = food_map[values[3]]
                food_format = food.format(foodVid, foodVid, values[3])

            provinceVid = -1
            if province_map.get(provinceName, -1) == -1:
                province_map[provinceName] = vid
                provinceVid = vid
                vid = vid + 1
            else:
                provinceVid = province_map[provinceName]

            province_format = province.format(provinceVid, provinceVid, provinceName)

            cityName = str(addr_dict["geocodes"][0]["city"])
            cityVid = -1
            if city_map.get(cityName, -1) == -1:
                city_map[cityName] = vid
                cityVid = vid
                vid = vid + 1
            else:
                cityVid = city_map[cityName]
            city_format = city.format(cityVid, cityVid, cityName)

            districtName = str(addr_dict["geocodes"][0]["district"])
            districtVid = -1
            if district_map.get(districtName, -1) == -1:
                district_map[districtName] = vid
                districtVid = vid
                vid = vid + 1
            else:
                districtVid = district_map[districtName]
            district_format = district.format(districtVid, districtVid, districtName)

            intro = clearText(values[6])
            introVid = -1
            intro_object = ""
            if intro != "":
                introVid = vid
                vid = vid + 1
                intro_object = object1.format(introVid, introVid, intro)

            imagesVid = -1
            images_object = ""
            images = clearText(values[7])
            if images != "":
                images = images.replace("\r\n", " ")
                imagesVid = vid
                vid = vid + 1
                images_object = object1.format(imagesVid, imagesVid, images)

            traffic = clearText(values[9])
            trafficVid = -1
            traffic_object = ""
            if traffic != "":
                trafficVid = vid
                vid = vid + 1
                traffic_object = object1.format(trafficVid, trafficVid, traffic)

            price = clearText(values2[6])
            priceVid = -1
            price_object = ""
            if price != "":
                priceVid = vid
                vid = vid + 1
                price_object = object1.format(priceVid, priceVid, price)

            foreign = clearText(values2[7])
            foreignVid = -1
            foreign_object = ""
            if foreign != "":
                foreignVid = vid
                vid = vid + 1
                foreign_object = object1.format(foreignVid, foreignVid, foreign)

            square = clearText(values2[8])
            squareVid = -1
            square_object = ""
            if square != "":
                squareVid = vid
                vid = vid + 1
                square_object = object1.format(squareVid, squareVid, square)

            famous = clearText(values2[10])
            famousVid = -1
            famous_object = ""
            if famous != "":
                famousVid = vid
                vid = vid + 1
                famous_object = object1.format(famousVid, famousVid, famous)

            # 单独处理
            climate = clearText(values2[13])
            climateVid = -1
            climate_object = ""
            if climate != "":
                if climate_map.get(climate, -1) == -1:
                    climate_map[climate] = vid
                    climateVid = vid
                    vid = vid + 1
                else:
                    climateVid = climate_map[climate]
                climate_object = object1.format(climateVid, climateVid, climate)

            openTime = clearText(values2[14])
            openTimeVid = -1
            openTime_object = ""
            if openTime != "":
                openTimeVid = vid
                vid = vid + 1
                openTime_object = object1.format(openTimeVid, openTimeVid, openTime)

            suggestionTime = clearText(values2[15])
            suggestionTimeVid = -1
            suggestionTime_object = ""
            if suggestionTime != "":
                suggestionTimeVid = vid
                vid = vid + 1
                suggestionTime_object = object1.format(suggestionTimeVid, suggestionTimeVid, suggestionTime)

            # 单独处理
            sightLevel = clearText(values2[16])
            sightLevelVid = -1
            sightLevel_object = ""
            if sightLevel != "":
                sightLevel = parseSightLevel(sightLevel)
                if level_map.get(sightLevel, -1) == -1:
                    level_map[sightLevel] = vid
                    sightLevelVid = vid
                    vid = vid + 1
                else:
                    sightLevelVid = level_map[sightLevel]
                sightLevel_object = object1.format(sightLevelVid, sightLevelVid, sightLevel)

            # 单独处理
            suggestionSeason = clearText(values2[17])
            suggestionSeasonVid = -1
            suggestionSeason_object = ""
            if suggestionSeason != "":
                suggestionSeasonVid = vid
                vid = vid + 1
                suggestionSeason_object = object1.format(suggestionSeasonVid, suggestionSeasonVid,
                                                         suggestionSeason)
            if provinceVid != -1:
                session.execute(province_format)

            if cityVid != -1:
                session.execute(city_format)
                relation_object = relation.format(sightVid, cityVid, eid, "城市")
                session.execute(relation_object)
                eid = eid - 1
                relation_object = relation.format(cityVid, provinceVid, eid, "省份")
                session.execute(relation_object)
                eid = eid - 1

            if sightVid != -1:
                session.execute(sight_format)
                if districtVid != -1:
                    session.execute(district_format)
                    relation_object = relation.format(districtVid, cityVid, eid, "城市")
                    session.execute(relation_object)
                    eid = eid - 1
                    relation_object = relation.format(sightVid, districtVid, eid, "区域")
                    session.execute(relation_object)
                    eid = eid - 1
                if introVid != -1:
                    session.execute(intro_object)
                    relation_object = relation.format(sightVid, introVid, eid, "景点介绍")
                    session.execute(relation_object)
                    eid = eid - 1

                if imagesVid != -1:
                    session.execute(images_object)
                    relation_object = relation.format(sightVid, imagesVid, eid, "景点图片")
                    session.execute(relation_object)
                    eid = eid - 1

                if trafficVid != -1:
                    session.execute(traffic_object)
                    relation_object = relation.format(sightVid, trafficVid, eid, "交通")
                    session.execute(relation_object)
                    eid = eid - 1

                if priceVid != -1:
                    session.execute(price_object)
                    relation_object = relation.format(sightVid, priceVid, eid, "门票价格")
                    session.execute(relation_object)
                    eid = eid - 1

                if foreignVid != -1:
                    session.execute(foreign_object)
                    relation_object = relation.format(sightVid, foreignVid, eid, "外文名")
                    session.execute(relation_object)
                    eid = eid - 1

                if squareVid != -1:
                    session.execute(square_object)
                    relation_object = relation.format(sightVid, squareVid, eid, "景点面积")
                    session.execute(relation_object)
                    eid = eid - 1

                if famousVid != -1:
                    session.execute(famous_object)
                    relation_object = relation.format(sightVid, famousVid, eid, "著名景点")
                    session.execute(relation_object)
                    eid = eid - 1

                if climateVid != -1:
                    session.execute(climate_object)
                    relation_object = relation.format(sightVid, climateVid, eid, "气候类型")
                    session.execute(relation_object)
                    eid = eid - 1

                if openTimeVid != -1:
                    session.execute(openTime_object)
                    relation_object = relation.format(sightVid, openTimeVid, eid, "开放时间")
                    session.execute(relation_object)
                    eid = eid - 1

                if suggestionTimeVid != -1:
                    session.execute(suggestionTime_object)
                    relation_object = relation.format(sightVid, suggestionTimeVid, eid, "游玩时间")
                    session.execute(relation_object)
                    eid = eid - 1

                if sightLevelVid != -1:
                    session.execute(sightLevel_object)
                    relation_object = relation.format(sightVid, sightLevelVid, eid, "景区级别")
                    session.execute(relation_object)
                    eid = eid - 1

                if suggestionSeasonVid != -1:
                    session.execute(suggestionSeason_object)
                    relation_object = relation.format(sightVid, suggestionSeasonVid, eid, "游玩季节")
                    session.execute(relation_object)
                    eid = eid - 1

            elif foodVid != -1:
                session.execute(food_format)

                if districtVid != -1:
                    session.execute(district_format)
                    relation_object = relation.format(districtVid, cityVid, eid, "城市")
                    session.execute(relation_object)
                    eid = eid - 1
                    relation_object = relation.format(foodVid, districtVid, eid, "区域")
                    session.execute(relation_object)
                    eid = eid - 1
                if introVid != -1:
                    session.execute(intro_object)
                    relation_object = relation.format(foodVid, introVid, eid, "美食介绍")
                    session.execute(relation_object)
                    eid = eid - 1

                if imagesVid != -1:
                    session.execute(images_object)
                    relation_object = relation.format(foodVid, imagesVid, eid, "美食图片")
                    session.execute(relation_object)
                    eid = eid - 1

                if trafficVid != -1:
                    session.execute(traffic_object)
                    relation_object = relation.format(foodVid, trafficVid, eid, "交通")
                    session.execute(relation_object)
                    eid = eid - 1

                if values[8] != "":
                    addrVid = -1
                    if addr_map.get(values[8], -1) == -1:
                        addr_map[values[8]] = vid
                        addrVid = vid
                        vid = vid + 1
                    else:
                        addrVid = addr_map[values[8]]
                    addr_object = object1.format(addrVid, addrVid, values[8])
                    session.execute(addr_object)
                    relation_object = relation.format(foodVid, addrVid, eid, "地址")
                    session.execute(relation_object)
                    eid = eid - 1

        for k in name_diff:
            f.write(k + " " + name_diff[k] + "\n")
    except:
        # 释放会话
        print("last vid = ", vid)
        print("last eid = ", eid)
        '''
        city_map = {}
        climate_map = {}
        district_map = {}
        food_map = {}
        level_map = {}
        province_map = {}
        sight_map = {}
        addr_map = {}
        '''
        print(city_map)
        print(climate_map)
        print(district_map)
        print(food_map)
        print(level_map)
        print(province_map)
        print(sight_map)
        print(addr_map)
        print(name_diff)
        session.release()
        connection_pool.close()
        cursor.close()
        f.close()

    print(len(sight_map))
    print(sight_map)