# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 10:49:52 2022

@author: rdi420

batch download newspaper trial
"""

import newspaper
from newspaper import news_pool
#%%
slate_paper = newspaper.build('http://slate.com')
tc_paper = newspaper.build('http://techcrunch.com')
espn_paper = newspaper.build('http://espn.com')

papers = [slate_paper, tc_paper, espn_paper]
news_pool.set(papers, threads_per_source=2) # (3*2) = 6 threads total
news_pool.join()

#%%
from newspaper import Article

a = Article('http://www.cnn.com/2014/01/12/world/asia/north-korea-charles-smith/index.html'
    , keep_article_html=True)

a.download()
a.parse()

a.article_html
#u'<div> \n<p><strong>(CNN)</strong> -- Charles Smith insisted Sunda...'


#%%

from newspaper import Source
import nltk
nltk.download('punkt')

cnn_paper = Source('https://www.theguardian.com/environment')

print(cnn_paper.size()) # no articles, we have not built the source

cnn_paper.build()
print(cnn_paper.size())

for article in cnn_paper.articles: 
    print(article.url)
#%%    
cnn_article = cnn_paper.articles[1]
cnn_article.download()
cnn_article.parse()
cnn_article.nlp()
print(cnn_article.title)
    
#%% 

#guardian_paper = Source('https://www.theguardian.com/environment/flooding')
guardian_paper = Source('https://www.theguardian.com/environment/flooding?page=7')
guardian_paper.build()
print(guardian_paper.size())

for article in guardian_paper.articles: 
    print(article.url)

#%% Get all web links/address on a page
import requests
from bs4 import BeautifulSoup as bs
  
#URL = 'https://www.geeksforgeeks.org/page/'
#URL = 'https://www.theguardian.com/environment/flooding?page='
URL = 'https://www.theguardian.com/world/2010/mar/15/all'

for page in range(1,10):
    # pls note that the total number of
    # pages in the website is more than 5000 so i'm only taking the
    # first 10 as this is just an example
  
    #req = requests.get(URL + str(page) )
    req = requests.get(URL)
    soup = bs(req.text, 'html.parser')
  
    #titles = soup.find_all('div',attrs={'class','head'})
    #title = soup.find_all('span', attrs = {'js-headlines-text'})
    #title = soup.find_all('a')
    titles = [link.get('href') for link in soup.find_all('a')]
    
    print(page)
    #titles[:3]
    
    for i in range(4,19):
        if page>1:
            print("({}-3)+{}*15".format(i, page) + titles[i].text)
        #else:
            #print(f"{i-3}" + titles[i].text)
    break

#%% Save text from news
import pickle # for the content/news in its freshes form
# key_list = ['earthquake', 'volcano', 'volcanic', 'tsunami', 'ash', 
#             'lahar', 'lava', 'pyroclastic', 'storm', 'derecho', 
#             'hail', 'lightning', 'thunder', 'rain', 'tornado', 
#             'sand', 'dust', 'blizzard', 'surge', 'wind', 'ice', 
#             'frost', 'freeze', 'avalanche', 'snow', 'debris', 
#             'landslide', 'seiche', 'drought', 'fire', 'glacial', 
#             'dam', 'disease', 'crisis', 'infestation', 'pyroclastic',
#             'hazard', 'disaster']
key_list = ['flood']

def check_keyword(text):
    judge = False
    for keyword in key_list:
        elem_to_find = keyword
        # element exists in list of lists or not?
        judge = any(elem_to_find in sublist.lower() for sublist in text)
        if judge == True:
            break
    return judge

def save_news_text(web_link, date): # Both variables are strings
    try:
        req = requests.get(web_link)
        page_info = bs(req.text, 'html.parser')
        narrow = page_info.find_all('p', {'class': "dcr-xry7m2"})
        if narrow != []:   
            text = [paragraph.get_text() for paragraph in narrow]
            judge = check_keyword(text)
            if judge == True:
                # for the guardian, the web link is always the last bit of the website address
                file_name = 'C:/MultiHazard/Data_Mining/guardian/'+ date +'_' + web_link.split('/')[-1] + '.pkl'
                with open(file_name, 'wb') as fp:
                    pickle.dump(text, fp)
    except:
        print("I SKIPPED " + web_link)        
            
            
#%% Open the pickled text file
path = 'C:/MultiHazard/Data_Mining/guardian/'
#path += '2021aug22_flooding-in-new-york-as-hurricane-henri-approaches-north-east-coast.pkl'
path += '2008sep16_naturaldisasters.flooding.pkl'

with open(path, 'rb') as f:
      mynewlist = pickle.load(f)

#%% Loop through day of a given year
# Each day has a search result
import requests
from bs4 import BeautifulSoup as bs
def days_of_year_loop(year):
    month = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    days = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
    for m in month:
        for day in days:
            print('DATE ' + m + day)
            url = 'https://www.theguardian.com/world/{}/{}/{}/all'.format(year, m, day)
            try:
                req = requests.get(url)
                soup = bs(req.text, 'html.parser')
                link_list0 = [link.get('href') for link in soup.find_all('a')]
                link_list = [i for i in link_list0 if i is not None]
                date = str(year) + m + day
                if link_list != []:
                    # save every piece of useful news
                    for link in link_list:
                        save_news_text(link, date)
            except:
                print("I SKIPPED " + url)
            print('\n')
                


        



