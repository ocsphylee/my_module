# coding = utf-8
'''
Author: Ocsphy
Date: 2019/5/21 13:31
'''
from selenium import webdriver
import time
import pymysql
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options

options = Options()
# options.add_argument('-headless')
driver = Firefox(options=options)
driver.get(href)
time.sleep(1)
# title = driver.find_element_by_class_name("report-title").text
# news = driver.find_element_by_class_name("newsContent").text

