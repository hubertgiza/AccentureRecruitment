from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
from requests import get

driver = webdriver.Chrome("/home/hubert/PycharmProjects/AccentureRecruitment/chromedriver")
URL = "https://relatedwords.org/relatedto/physical%20fitness"
driver.get(URL)
driver.quit()
