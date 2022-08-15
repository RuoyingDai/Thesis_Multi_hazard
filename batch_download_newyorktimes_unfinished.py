# -*- coding: utf-8 -*-
"""
Created on Wed May 18 13:29:07 2022

@author: rdi420
"""


import requests
from bs4 import BeautifulSoup as bs
url = "https://www.nytimes.com/search?dropmab=false&endDate=20210714&query=&sort=best&startDate=20210714"
req = requests.get(url)
soup = bs(req.text, 'html.parser')
link_list0 = [link.get() for link in soup.find_all('p', {'class':'css-16nhkrn'})]
link_list = [i for i in link_list0 if i is not None]


