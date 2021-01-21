#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 14:19:47 2021

@author: jzimmer1
"""

from bs4 import BeautifulSoup
import os
import json
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download("stopwords")
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
from nltk.stem import 	WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

# === === Helper functions === ===
def write_json(data, filename):
    with open(filename,"w") as f:
        json.dump(data,f)
    return None
def get_json(filename):
    with open(filename) as f:
        jsonobj = json.load(f)
    return jsonobj
    
# === === === === === === === ===


class Text():
    def __init__(self):
        self.filemap = get_json("q_filemap.json")
        self.t0, self.t1 = self.go_thru()
        #write_json(self.t0,"uncleanqt0.json")
        #write_json(self.t1,"uncleanqt1.json")
        
    def get_text_from_website(self,filename):
        soup = BeautifulSoup(open(filename,encoding="ISO-8859-1"),features="lxml")
        allps = soup.find_all('p')
        alltext = []
        for p in allps:
            for i in word_tokenize(p.get_text()):
                alltext.append(i)
        return alltext
    
    def go_thru(self):
        tokenlist0 = []
        tokenlist1 = []
        for f in self.filemap:
            text = self.get_text_from_website(f)
            if self.filemap[f] == 0:
                for i in text:
                    tokenlist0.append(i)
            else:
                for i in text:
                    tokenlist1.append(i)
        return tokenlist0,tokenlist1
    
Text() 