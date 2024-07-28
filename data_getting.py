from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import re
import pandas as pd

chrome_options = Options()
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--log-level=3")
chrome_options.add_argument("--ignore-certificate-errors")
chrome_options.add_argument("--disable-extensions")
chrome_options.add_argument("--disable-software-rasterizer")

service = Service('C:/Users/yarom/work_study/chromedriver/chromedriver.exe')

driver = webdriver.Chrome(service=service, options=chrome_options)

def login_to_telegram():
    driver.get("https://web.telegram.org/")
    time.sleep(15)
    driver.get('https://google.com')

def get_posts(channel_url):
    driver.get(channel_url)
    time.sleep(10)

    posts = set()
    for i in range(12): # количество скроллов, которые мы сделаем (по хорошему бы поставить побольше)
        elements = driver.find_elements(By.CLASS_NAME, 'text-content.clearfix.with-meta')
        for element in elements:
            posts.add(element.text)
        driver.execute_script('document.getElementsByClassName("MessageList custom-scroll no-avatars no-composer with-default-bg scrolled")[0].scrollBy(0,-2500)')
        time.sleep(3)
    
    return posts

login_to_telegram()
z = get_posts('https://web.telegram.org/a/#-1001763496063')
print(len(z))

pd.DataFrame(list(z), columns=['posts']).to_excel('data.xlsx', index=False)