# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 09:37:07 2022

@author: rdi420

# extract location in text
"""

import spacy
from spacy import displacy 
#import en_core_web_sm
# I need to download the pipeline below firstly in the console.
# python -m spacy download en_core_web_sm

nlp = spacy.load("en_core_web_sm")
# Text with nlp
doc = nlp(" Multiple tornado warnings were issued for parts of New York on Sunday night.The first warning, which expired at 9 p.m., covered the Bronx, Yonkers and New Rochelle. More than 2 million people live in the impacted area.")
# Display Entities
displacy.render(doc, style="ent")
#%%
#
nlp_wk = spacy.load("xx_ent_wiki_sm") # don't forget ot download the package/pipeline through console
doc = nlp_wk("Multiple tornado warnings were issued for parts of New York on Sunday night.The first warning, which expired at 9 p.m., covered the Bronx, Yonkers and New Rochelle. More than 2 million people live in the impacted area.")
displacy.render(doc, style="ent")