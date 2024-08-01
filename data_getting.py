from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import re
import pandas as pd



def login_to_telegram():
    global driver

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

    driver.get("https://web.telegram.org/")
    time.sleep(12)
    driver.get('https://google.com')

def get_posts(channel_url, amount_of_posts, type_of_url):
    '''
    type_of_url : если вы подписчик канала, то 1, иначе 2
    '''
    driver.get(channel_url)
    time.sleep(15)
    print('started')
    posts = set()
    while len(posts) < amount_of_posts:
        print(len(posts))
        elements = driver.find_elements(By.CLASS_NAME, 'text-content.clearfix.with-meta')
        for element in elements:
            posts.add(element.text)
        if type_of_url == 1:
            driver.execute_script('document.getElementsByClassName("MessageList custom-scroll no-avatars no-composer with-default-bg scrolled")[0].scrollBy(0,-4500)')
        elif type_of_url == 2:
            driver.execute_script('document.getElementsByClassName("MessageList custom-scroll no-avatars no-composer with-bottom-shift with-default-bg")[0].scrollBy(0,-4500)')
        time.sleep(2)

    
    return posts

def posts_to_excel(z, file_name):
    pd.DataFrame(list(z), columns=['posts']).to_excel(file_name, index=False)

# document.getElementsByClassName("MessageList custom-scroll no-avatars no-composer with-default-bg scrolled")[0].scrollBy(0,-4500) - альтернатвный вариант (на каждом канале по-разному)