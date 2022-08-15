# -*- coding: utf-8 -*-
"""
Created on Wed May  4 15:41:52 2022

@author: rdi420
"""

  

#%% Save text from news
import requests
from bs4 import BeautifulSoup as bs
import pickle # for the content/news in its freshes form
key_list = ['earthquake', 'volcano', 'volcanic', 'tsunami', 'ash', 
            'lahar', 'lava', 'pyroclastic', 'storm', 'derecho', 
            'hail', 'lightning', 'thunder', 'rain', 'tornado', 
            'sand', 'dust', 'blizzard', 'surge', 'wind', 'ice', 
            'frost', 'freeze', 'avalanche', 'snow', 'debris', 
            'landslide', 'seiche', 'drought', 'fire', 'glacial', 
            'dam', 'disease', 'crisis', 'infestation', 'pyroclastic',
            'hazard', 'disaster', 'flood']

def check_keyword(text):
    judge = False
    for keyword in key_list:
        elem_to_find = keyword
        # element exists in list of lists or not?
        judge = any(elem_to_find in sublist.lower() for sublist in text)
        if judge == True:
            print(keyword)
            break
    return judge

def save_news_text(web_link, date): # Both variables are strings
    try:
        req = requests.get(web_link)
        page_info = bs(req.text, 'html.parser')
#        narrow = page_info.find_all('p', {'class': "dcr-xry7m2"})
        narrow = page_info.find_all('p')
        if narrow != []:   
            text0 = [paragraph.get_text() for paragraph in narrow]
            text = [piece for piece in text0 if len(piece)>3]
            judge = check_keyword(text)
            if judge == True:
                # for the guardian, the web link is always the last bit of the website address
                file_name = 'C:/MultiHazard/Data_Mining/aljazeera/'+ date +'_' + web_link.split('/')[-1] + '.pkl'
                with open(file_name, 'wb') as fp:
                    pickle.dump(text, fp)
    except:
        print("I SKIPPED " + web_link)        
            
            
#%% Open the pickled text file
path = 'C:/MultiHazard/Data_Mining/aljazeera/'
#path += '2021aug22_flooding-in-new-york-as-hurricane-henri-approaches-north-east-coast.pkl'
path += '2003-4-20_iran-struggles-to-keep-stance-of-active-neutrality.pkl'
with open(path, 'rb') as f:
      mynewlist = pickle.load(f)
#%% split sentences
# https://stackoverflow.com/questions/4576077/how-can-i-split-a-text-into-sentences
import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences
#%% Loop through day of a given year
# Each day has a search result
import requests
from bs4 import BeautifulSoup as bs
def days_of_year_loop(year):
    for m in range(1, 13):
        for day in range(1, 32):
            print('DATE ' + str(m) + str(day))
            url = 'https://www.aljazeera.com/search/{}-{}-{}'.format(str(year), m, day)
            try:
                req = requests.get(url)
                soup = bs(req.text, 'html.parser')
                link_list0 = [link.get('href') for link in soup.find_all('a')]
                link_list = [i for i in link_list0 if i is not None]
                date = '{}-{}-{}'.format(str(year) ,m, day)
                trace = '{}/{}/{}'.format(str(year), m, day)
                if link_list != []:
                    # save every piece of useful news
                    for link in link_list:
                        if trace in link:
                            save_news_text(link, date)
            except:
                print("I SKIPPED " + url)
            print('\n')
                
#%%


days_of_year_loop(2015)
days_of_year_loop(2014)
days_of_year_loop(2013)
days_of_year_loop(2012)
days_of_year_loop(2011)
days_of_year_loop(2010)
days_of_year_loop(2009)
days_of_year_loop(2008)
days_of_year_loop(2007)
days_of_year_loop(2006)
days_of_year_loop(2005)
days_of_year_loop(2004)
days_of_year_loop(2003)
days_of_year_loop(2002)
days_of_year_loop(2001)
days_of_year_loop(2000)


        



