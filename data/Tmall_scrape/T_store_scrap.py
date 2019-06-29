# coding = utf-8
'''
Author: Ocsphy
Date: 2019/6/2 19:46
'''

from bs4 import BeautifulSoup as bs
from selenium import webdriver
import time
import pandas as pd
from datetime import datetime


options = webdriver.ChromeOptions()
options.add_argument('user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36"')
options.add_experimental_option('excludeSwitches', ['enable-automation'])
driver = webdriver.Chrome(chrome_options=options)
