#!/usr/bin/env python
# coding: utf-8


import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdate
import matplotlib.ticker as mtick
import os
from openpyxl import load_workbook

#  load data, and seperate GMV, Sales, and ATV


def load_data(path, unit=1, index=None, growpby=None):
    """
    导入数据，并分别生成gmv,sales和atv三个数据表
    """
    data = pd.read_excel(path, index_col=index)
    data['gmv'] = data['gmv'] / unit
    data['sale_qtty'] = data['sale_qtty'] / unit
    timedata = data.groupby(growpby)
    new_data = timedata.sum()
    gmv = new_data['gmv'].unstack()
    sales = new_data['sale_qtty'].unstack()
    atv = gmv / sales
    return gmv, sales, atv


# load brands or catelog list


def load_cat(path, name):
    """
    导入品牌目录，并根据品类（name）获取品牌列表
    """
    brands = pd.read_excel(path).fillna(0)
    brand_lit = brands[name].tolist()
    while 0 in brand_lit:
        brand_lit.remove(0)
    return brand_lit


# calculate the growth


def yoy_growth(df, period):
    """计算yoy增长率
    """
    tmp = df.to_period(period).groupby('dt').sum()
    tmp1 = tmp['2018']
    tmp2 = tmp['2019']
    growth = tmp2.copy()
    m, n = tmp2.shape
    for i in range(n):
        for j in range(m):
            growth.iloc[j, i] = (tmp2.iloc[j, i] / tmp1.iloc[j, i] - 1)
    return growth


def cr_growth(df, period):
    """
    计算环比增长率
    """
    tmp = df.to_period(period).groupby('dt').sum()
    growth = tmp.copy()
    m, n = tmp.shape
    for i in range(n):
        growth.iloc[:, i] = (tmp.iloc[:, i] / tmp.iloc[:, i].shift(1) - 1)
    return growth


#  save datas


def bubble_data(gmv, date, kind=None, cat_gmv=None, brands=None):
    """
        格式化需要画泡泡图的数据
    input:  gmv（df）：需要画泡泡图的数据
            date(str): 日期 ‘2019-04-01’
            kind(str): 数据的类型（cat = '分品类'，其他表示具体的品牌数据）
            cat_gmv(df): 品类的总数据；当cat为品牌时，必选。
            brands(str)：品类的名称，当cat为品牌时，必选。
    output: bubble(df): 返回泡泡数据，份额、增速和gmv
    """

    if kind == "大家电" or kind == "小家电":
        kind = '家用电器'
    if kind == "cat":
        data = gmv.loc[date, :]
        total = data.sum()
        gmv_pct = data / total
        gmv_yoy = yoy_growth(gmv, "M").loc[date, :]
    else:
        total = cat_gmv.loc[date, kind]
        data = gmv[brands].loc[date, :]
        gmv_pct = data / total
        gmv_yoy = yoy_growth(gmv, "M")[brands].loc[date, :]
    bubble = pd.DataFrame(gmv_pct)
    bubble.columns = ['平均市场份额']
    bubble["销售额增速"] = np.array(gmv_yoy)
    bubble["销售额"] = np.array(data)
    return bubble


def save_bubble(gmv, cat_gmv, catalog, catlist, date, bubblepath, file_path):
    """
    保存bubble数据
    """
    writer = pd.ExcelWriter(file_path + date + bubblepath)
    cat_bubble = bubble_data(cat_gmv, date, kind='cat')
    cat_bubble.to_excel(writer, sheet_name='平台')
    for kind in catlist:
        brands = load_cat(catalog, kind)
        b_brand = bubble_data(
            gmv,
            date,
            kind=kind,
            cat_gmv=cat_gmv,
            brands=brands)
        b_brand.to_excel(writer, sheet_name=kind)
    writer.save()
    writer.close()


def save_cat_sheet(df, path, name):
    '''保存品类数据
    input: df（df）: 需要整理的原始数据
            path(str)：保存的路径
            name: sheet 名
    '''
    n = df.shape[0]
    my_growth = yoy_growth(df, 'M')
    qy_growth = yoy_growth(df, "Q")
    c_growth = cr_growth(df, "M")

    workbook = load_workbook(path)
    writer = pd.ExcelWriter(path, engine='openpyxl')
    writer.book = workbook
    if name in workbook.sheetnames:
        workbook.remove(workbook[name])
        workbook.save(path)
    df.to_excel(writer, sheet_name=name)
    my_growth.to_excel(writer, sheet_name=name, startrow=n + 3)
    qy_growth.to_excel(writer, sheet_name=name, startrow=2 * n + 3)
    c_growth.to_excel(writer, sheet_name=name, startrow=3 * n + 3)
    writer.save()
    writer.close()





def save_brand(cat, path, cat_path, gmv, sales, atv, gmv_g, sales_g, atv_g):
    '''保存品牌数据
    input: cat（str）: 需要整理的品类的名称
            path(str)：保存的路径
    '''
    brands = load_cat(cat_path, cat)
    gmv = gmv[brands]
    sales = sales[brands]
    atv = atv[brands]
    gmv_g = gmv_g[brands]
    sales_g = sales_g[brands]
    atv_g = atv_g[brands]

    m, n = gmv.shape
    workbook = load_workbook(path)
    writer = pd.ExcelWriter(path, engine='openpyxl')
    writer.book = workbook
    if cat in workbook.sheetnames:
        workbook.remove(workbook[cat])
        workbook.save(path)

    gmv.to_excel(writer, sheet_name=cat)
    gmv_g.to_excel(writer, sheet_name=cat, startcol=n + 3)

    sales.to_excel(writer, sheet_name=cat, startrow=m + 3)
    sales_g.to_excel(writer, sheet_name=cat, startrow=m + 3, startcol=n + 3)

    atv.to_excel(writer, sheet_name=cat, startrow=2 * m + 6)
    atv_g.to_excel(writer, sheet_name=cat, startrow=2 * m + 6, startcol=n + 3)
    writer.save()
    writer.close()

def format_data(brand_path, cat_path,cat_name_path,bubblepath,catlist,plat,date,unit,bubble = True):

    cat_gmv, cat_sales, cat_atv = load_data(
        cat_path, index='dt', growpby=[
            'dt', 'cid1_name'], unit=unit)
    gmv, sales, atv = load_data(
        brand_path, index='dt', growpby=[
            'dt', 'main_brand_name'], unit=unit)
    print("----导入数据----")

    # 2. 计算增长率
    gmv_g = yoy_growth(gmv, "M")
    sales_g= yoy_growth(sales, "M")
    atv_g = yoy_growth(atv, "M")
    print("----增长率----")

    # 3. 保存品类数据
    save_cat_sheet(cat_gmv, cat_path, 'gmv')
    save_cat_sheet(cat_sales, cat_path, 'sales')
    save_cat_sheet(cat_atv, cat_path, 'atv')
    print("----品类----")

    # 4. 保存品牌数据
    for cat in catlist:
        save_brand(
            cat,
            brand_path,
            cat_name_path,
            gmv,
            sales,
            atv,
            gmv_g,
            sales_g,
            atv_g)
    print("----品牌----")
    # 5. 保存泡泡图格式数据

    if bubble:
        save_bubble(
            gmv,
            cat_gmv,
            cat_name_path,
            catlist,
            date,
            bubblepath,
            plat)
    print("----完成----")

# * 运行
if __name__ == '__main__':

    unit = 100000000
    date = '2019-05-01'
    brand_path_T = "./JD/JD_brands.xlsx"
    cat_path_T = './JD/JD_cats.xlsx'
    catlist_T = ['医药保健', '酒类', '大家电', '小家电', '美妆个护', '服装鞋包']
    catlist_D = ['医药保健', '酒类', '大家电', '小家电', '食品饮料及生鲜']
    cat_name_path_T = './JD/JD_catalog.xlsx'
    bubblepath_T = 'bubble_data_JD.xlsx'
    plat_T = "./JD/"
    format_data(brand_path_T, cat_path_T, cat_name_path_T, bubblepath_T, catlist_D, plat_T,date,unit)



    ''' 处理天猫数据'''
    '''
    # 1.导入品类和品牌的原始数据
    brand_path_T = "./Tmall/Tmall_brand.xlsx"
    cat_path_T = './Tmall/Tmall_cat.xlsx'
    cat_gmv_T, cat_sales_T, cat_atv_T = load_data(
        cat_path_T, index='dt', growpby=[
            'dt', 'cid1_name'], unit=unit)
    gmv_T, sales_T, atv_T = load_data(
        brand_path_T, index='dt', growpby=[
            'dt', 'main_brand_name'], unit=unit)

    # 2. 计算增长率
    gmv_g_T = yoy_growth(gmv_T, "M")
    sales_g_T = yoy_growth(sales_T, "M")
    atv_g_T = yoy_growth(atv_T, "M")

    # 3. 保存品类数据

    save_cat_sheet(cat_gmv_T, cat_path_T, 'gmv')
    save_cat_sheet(cat_sales_T, cat_path_T, 'sales')
    save_cat_sheet(cat_atv_T, cat_path_T, 'atv')

    # 4. 保存品牌数据
    catlist_T = ['医药保健', '酒类', '大家电', '小家电', '美妆个护', '服装鞋包']
    cat_name_path_T = './Tmall/brand_catalog.xlsx'
    for cat in catlist_T:
        save_brand(
            cat,
            brand_path_T,
            cat_name_path_T,
            gmv_T,
            sales_T,
            atv_T,
            gmv_g_T,
            sales_g_T,
            atv_g_T)

    # 5. 保存泡泡图格式数据
    bubblepath_T = 'bubble_data_T.xlsx'
    save_bubble(
        gmv_T,
        cat_gmv_T,
        cat_name_path_T,
        catlist_T,
        date,
        bubblepath_T,
        "./Tmall/")
    '''
    '''--------------------------------------------------------------'''
    '''处理京东数据'''
    '''
    # 1.导入品类和品牌的原始数据
    brand_path_D = "./JD/JD_brands.xlsx"
    cat_path_D = './JD/JD_cats.xlsx'
    cat_gmv_D, cat_sales_D, cat_atv_D = load_data(
        cat_path_D, index='dt', growpby=[
            'dt', 'cid1_name'], unit=unit)
    gmv_D, sales_D, atv_D = load_data(
        brand_path_D, index='dt', growpby=[
            'dt', 'main_brand_name'], unit=unit)

    # 2. 计算增长率
    gmv_g_D = yoy_growth(gmv_D, "M")
    sales_g_D = yoy_growth(sales_D, "M")
    atv_g_D = yoy_growth(atv_D, "M")

    # 3. 保存品类数据

    save_cat_sheet(cat_gmv_D, cat_path_D, 'gmv')
    save_cat_sheet(cat_sales_D, cat_path_D, 'sales')
    save_cat_sheet(cat_atv_D, cat_path_D, 'atv')

    # 4. 保存品牌数据
    catlist_D = ['医药保健', '酒类', '大家电', '小家电', '食品饮料及生鲜']
    cat_name_path_D = './JD/JD_catalog.xlsx'
    for cat in catlist_D:
        save_brand(
            cat,
            brand_path_D,
            cat_name_path_D,
            gmv_D,
            sales_D,
            atv_D,
            gmv_g_D,
            sales_g_D,
            atv_g_D)

    # 5. 保存泡泡图格式数据
    bubblepath_D = 'bubble_data_JD.xlsx'
    save_bubble(
        gmv_D,
        cat_gmv_D,
        cat_name_path_D,
        catlist_D,
        date,
        bubblepath_D,
        "./JD/")
    '''