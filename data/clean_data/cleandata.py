# coding = utf-8
'''
Author: Ocsphy
Date: 2019/6/29 13:22
'''
import numpy as np
import pandas as pd


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

class DFData():
    """
    读取和保存数据
    """
    def __init__(self, path, unit=1, index=None, groupby=None):

        data = pd.read_excel(path, index_col=index)
        data['gmv'] = data['gmv'] / unit
        data['sale_qtty'] = data['sale_qtty'] / unit

        self.data = data.groupby(groupby).sum()
        self.gmv = self.data['gmv'].unstack()
        self.sales = self.data['sale_qtty'].unstack()
        self.atv = self.gmv / self.sales

        if 'source' in data.columns and "cat" in path:
            groupby.insert(1,'source')
            self.cid_data = data.groupby(groupby).sum()
            self.pop_gmv = self.cid_data.loc[(slice(None),'pop'),'gmv'].unstack().reset_index(level='source',drop=True)
            self.self_gmv = self.cid_data.loc[(slice(None), 'self'), 'gmv'].unstack().reset_index(level='source',drop=True)
        else:
            self.cid_data = None
            self.pop_gmv = None
            self.self_gmv = None

unit = 100000000
JD = DFData("./JD/JD_brands.xlsx", index='dt', groupby=['dt', 'main_brand_name'], unit=unit)
TM = DFData("./Tmall/Tmall_brand.xlsx", index='dt', groupby=['dt', 'main_brand_name'], unit=unit)
gmv = JD.gmv+ TM.gmv
gmv = gmv.dropna(axis=1,how='any')
brand = ['九阳','科沃斯','美的','苏泊尔','飞科']

print(gmv[brand])




