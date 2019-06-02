# coding = utf-8
'''
Author: Ocsphy
Date: 2019/5/21 13:31
'''

import urllib.request
import urllib.parse
from bs4 import BeautifulSoup as bs
import re
from selenium import webdriver
import time

def get_html(url):
    headers = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36',
              'Cookie': 'cna=+81+FNQhPFICAWXkoli7LlUp; isg=BFJSBx9TWgP_D6Yemta7pTX6oBj0y1eVJiCT4hyrJIXwL_IpBPEuDVpNm0u2RM6V; l=bBMNk-Hgv-jlgIjtKOCwZuI8Ls7TLIOYYuPRwC0Xi_5CO_T_gi_OlHS_FE96Vs5R6YLB4KXJ7Lp9-etki; t=be1614ea16cf7575e56eaedfb655a430; _tb_token_=78389baed73e9; cookie2=1578ab51af902cfa6bf0d1ca52894447; sm4=310100; _m_h5_tk=33af4827cd363c6d788c17bea8a61bd7_1559128200105; _m_h5_tk_enc=af49622ccd35d88b9f8d591e46f3f3ea; pnm_cku822=; cq=ccp%3D0; hng=""; uc1=cookie16=W5iHLLyFPlMGbLDwA%2BdvAGZqLg%3D%3D&cookie21=UtASsssme%2BBq&cookie15=VT5L2FSpMGV7TQ%3D%3D&existShop=true&pas=0&cookie14=UoTZ7Yt0kZ7zrg%3D%3D&tag=8&lng=zh_CN; uc3=vt3=F8dBy3vNDyLaZK4eF2I%3D&id2=WvNB%2Bbe2lplz&nk2=F5RAQIgeUP0StXNf&lg2=W5iHLLyFOGW7aA%3D%3D; tracknick=tb5277119_88; _l_g_=Ug%3D%3D; ck1=""; unb=931467101; lgc=tb5277119_88; cookie1=AVxDAGeNrwdWlRHQ%2FfRyZ6GcpkAQVmC6TpXjKHOjseQ%3D; login=true; cookie17=WvNB%2Bbe2lplz; _nk_=tb5277119_88; uss=""; csg=4196d569; skt=7201f084b0853f2b; otherx=e%3D1%26p%3D*%26s%3D0%26c%3D0%26f%3D0%26g%3D0%26t%3D0; swfstore=84117; whl=-1%260%260%260; x=__ll%3D-1%26_ato%3D0'}
    req = urllib.request.Request(url,headers = headers)
    response = urllib.request.urlopen(req)
    content = response.read().decode(encoding='GBK')
    html = bs(content,features="lxml")
    return html

def get_cat(html):
    a = html.find_all('a', {"class": "cat-name fst-cat-name"})
    url_dic = {}
    for item in a[3:]:
        text = item.text.strip('【】')
        url = 'https:' + item['href']
        url_dic[text] = url
    return url_dic

if __name__ == '__main__':

    url = "https://nanjiren.tmall.com/category.htm?"
    html = get_html(url)
    url_dic = get_cat(html)
    print(html)

