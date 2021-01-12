#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 14:54:31 2021

@author: jzimmer1
"""

from bs4 import BeautifulSoup
import os
import json
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

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

class CamelotConfTranscript():
    def __init__(self):
        pass
    def get_transcript_text(self, DirName):
        allfiles = os.listdir(DirName)
        alltext = []
        for f in allfiles:
            soup = BeautifulSoup(open(DirName+f,encoding="ISO-8859-1"),features="lxml")
            allps = soup.find_all('p',class_="style7")
            for p in allps:
                alltext.append(p.get_text())
        return alltext
    
class ComparisonTranscript():
    def __init__(self, DirName):
        self.DirName = DirName
    def get_transcript_text_tedx(self, filename):
        soup = BeautifulSoup(open(self.DirName+filename,encoding="ISO-8859-1"),features="lxml")
        allps = soup.find_all('p')
        alltext = []
        for p in allps:
            alltext.append(p.get_text())
        return alltext
    def get_all_tedx(self):
        allfiles = os.listdir(self.DirName)
        alltext = []
        for f in allfiles:
            alltext += self.get_transcript_text_tedx(f)
        return alltext
    
class ComparisonBetween():
    def __init__(self, comparisonObj, comparisonObjTxt):
        self.it = CamelotConfTranscript()
        self.compareit = comparisonObj
        self.it_txt = self.it.get_transcript_text("ProjectCamelotConference/")
        self.compare_txt = comparisonObjTxt
    def make_df(self):
        df = pd.DataFrame()
        for p in self.it_txt:
            #df2 = pd.DataFrame([[p,1]],columns=list('TextClass'))
            print(type(p))
            #df2.head()
        return None
        


compareit = ComparisonTranscript("PCCComparison/")
rawtext = compareit.get_all_tedx()

doit = ComparisonBetween(compareit, rawtext)
doit.make_df()


# train = rawtext

# vectorizer = CountVectorizer(stop_words="english", max_features=100)
# X_train = vectorizer.fit_transform(train)
# #Y_train = train["spam"]
# train_vocab = vectorizer.get_feature_names()
# print(train_vocab)

