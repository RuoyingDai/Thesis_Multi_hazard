# -*- coding: utf-8 -*-
"""
Created on Thu May  5 18:03:39 2022

@author: rdi420
"""


#%%


# sentences1 = ['Hail stones the size of tennis balls were reported.',
# 'A tornado has swept through several villages in the Czech Republic, killing five people and leaving more than 150 others injured.',
# 'Heavy rain has once again caused deadly flooding in southern France.']

#sentences2 = ['Natural Gas may have been the cause of an explosion that killed at least two and injured six.',
#         'Eight people have been killed across the Pacific coast of South America after heavy rain caused flooding and landslides.',
#         'Officials mused that the rocket had apparently “gone off course” and made conciliatory noises.']

sentences1 = ['This follows the devastating floods in the region a week ago where at least 5 people died.',
'Severe flooding in southeastern France has left at least 4 people dead with 1 person still missing. ',
'As of 2 January, several forest fires devastate northern Corsica, affected by the storm Eleanor and its strong winds.',
'As many as 800 people had to be evacuated after the Eyjafjallajokull volcano in southern Iceland erupted yesterday. ',
'The eruption from within a glacier melted ice and triggered flooding in the surrounding area.',
'An overnight earthquake, triggered by Mount Etna\'s eruption two days ago, caused injuries and damage in Eastern Sicily early Wednesday morning',
'Ike is the third storm to smash into the coastal US this month, while large parts of the midwest were deluged when the Mississippi flooded.'
]

#sentences2 = ['Officials mused that the rocket had apparently “gone off course” and made conciliatory noises.'for i in range(len(sentences1))]
sentences2 = ['Queensland is enduring its second flood emergency in three months, with several regions on alert.'for i in range(len(sentences1))]

#Compute embedding for both lists
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)

#Compute cosine-similarits
cosine_scores = util.cos_sim(embeddings1, embeddings2)

#Output the pairs with their score
for i in range(len(sentences1)):
    print("s1:{} \ns2:{} \nScore: {:.4f}\n".format(sentences1[i], sentences2[i], cosine_scores[i][i]))
    
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




#%% Read in the newspaper pkl one by one 
# and then keep the newspaper that contains sentences
# which are highly similar to our collection

import pickle # for the content/news in its freshes form
import os


def check_news(f, filename):# address for one news file
    reference = ['This follows the devastating floods in the region a week ago where at least 5 people died.',
    'Severe flooding in southeastern France has left at least 4 people dead with 1 person still missing. ',
    'As of 2 January, several forest fires devastate northern Corsica, affected by the storm Eleanor and its strong winds.',
    'As many as 800 people had to be evacuated after the Eyjafjallajokull volcano in southern Iceland erupted yesterday. ',
    'The eruption from within a glacier melted ice and triggered flooding in the surrounding area.',
    'An overnight earthquake, triggered by Mount Etna\'s eruption two days ago, caused injuries and damage in Eastern Sicily early Wednesday morning',
    'Ike is the third storm to smash into the coastal US this month, while large parts of the midwest were deluged when the Mississippi flooded.'
    ]
    dest = 'C:/MultiHazard/Data_Mining/aljazeera2/' # new folder for selected news
    address = f + filename
    # Open the pickled text file
    with open(address, 'rb') as f:
          news = pickle.load(f)
    done = False
    for para in news:
        # ss for a set of sentences
        ss = split_into_sentences(para)
        for s in ss:

            #Compute embedding for both lists
            embeddings1 = model.encode(reference, convert_to_tensor=True)
            embeddings2 = model.encode(s, convert_to_tensor=True)
            
            #Compute cosine-similarits
            cosine_scores = util.cos_sim(embeddings1, embeddings2)

            if sum(cosine_scores>0.4)!=0:
                new = dest + filename
                with open(new, 'wb') as fp:
                    pickle.dump(news, fp)
                done = True
                print(filename)
                break
        if done == True:
            break
#%% 

# index = [idx for idx, s in enumerate(all_news) if '2011-8-15' in s][0]
# all_news = all_news[24941]               
# f = 'C:/MultiHazard/Data_Mining/aljazeera/' # f for folder
# all_news = os.listdir(f)
# #%%
# for item in all_news:
#     check_news(f, item)
#%%
#Output the pairs with their score
# for i in range(len(sentences1)):
#     print("s1:{} \ns2:{} \nScore: {:.4f}\n".format(sentences1[i], sentences2[i], cosine_scores[i][i]))
    
#%% for CNN
#https://transcripts.cnn.com/date/2001-12-30   

#%%
def main():
    print("Hello World! from Spdyer3.8")
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer('all-MiniLM-L6-v2')
#    f = 'C:/MultiHazard/Data_Mining/aljazeera/' # f for folder
    f = '~/Data_Mining/aljazeera/' # f for folder
    all_news = os.listdir(f)
    index = [idx for idx, s in enumerate(all_news) if '2011-8-15' in s][0]
    all_news = all_news[24941]               
    for item in all_news:
        check_news(f, item)

if __name__ == "__main__":
    main()

   
        
        



