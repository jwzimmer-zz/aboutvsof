#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 14:54:31 2021

@author: jzimmer1
"""

from bs4 import BeautifulSoup
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

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
                pdict = {"File":f, "Body":p.get_text(), "Class":1}
                alltext += [pdict]
        return alltext
    
class ComparisonTranscript():
    def __init__(self, DirName, contentName="Ted"):
        self.DirName = DirName
        self.contentName = contentName
    def get_transcript_text_tedx(self, filename):
        soup = BeautifulSoup(open(self.DirName+filename,encoding="ISO-8859-1"),features="lxml")
        allps = soup.find_all('p')
        alltext = []
        for p in allps:
            pdict = {"File":filename, "Body":p.get_text(), "Class":0}
            alltext += [pdict]
        return alltext
    def get_all_text(self):
        allfiles = os.listdir(self.DirName)
        alltext = []
        for f in allfiles:
            if self.contentName == "Ted":
                alltext += self.get_transcript_text_tedx(f)
            else:
                pass
        return alltext
    
class ComparisonBetween():
    def __init__(self, it, itTxt, comparisonObj, comparisonObjTxt):
        self.it = it
        self.compareit = comparisonObj
        self.it_txt = itTxt
        self.compare_txt = comparisonObjTxt
        self.df = self.make_df(self.it_txt, self.compare_txt)
    def make_df(self, list1, list2):
        df = pd.DataFrame(columns=['File','Body','Class'])
        #print(df.head())
        pccdf = pd.DataFrame(list1)
        comparedf = pd.DataFrame(list2)
        #print(pccdf.head())
        #print(comparedf.head())
        newdf = df.append(pccdf,ignore_index=True)
        newnewdf = newdf.append(comparedf,ignore_index=True)
        #print(newnewdf.head())
        return newnewdf
    
def get_simple_test_set():
    
    compareit = ComparisonTranscript("PCCComparison/")
    rawtext = compareit.get_all_text()
    it = CamelotConfTranscript()
    ittxt = it.get_transcript_text("ProjectCamelotConference/")

    doit = ComparisonBetween(it, ittxt, compareit, rawtext)
    #doit.df has all the training data in it
    
    testdf = pd.DataFrame(columns=["File","Body","Class"])
    test_pc = CamelotConfTranscript()
    pc_txt = test_pc.get_transcript_text("Test/PC/")
    test_ted = ComparisonTranscript("Test/Ted/")
    ted_txt = test_ted.get_all_text()
    
    testit = ComparisonBetween(test_pc,pc_txt,test_ted,ted_txt)
    #testit.df has all the test data in it

    train, val = train_test_split(doit.df, test_size=0.2, random_state=42)
    #print(train)

    vectorizer = CountVectorizer(stop_words="english", max_features=10000)
    X_train = vectorizer.fit_transform(train["Body"])
    Y_train = train["Class"]
    Y_train=Y_train.astype('int')
    train_vocab = vectorizer.get_feature_names()
    #print(train_vocab)
    
    #print([x for x in doit.df["Class"] if x not in (1,0)])
#=============================================================================
    model = LogisticRegression().fit(X_train,Y_train)
    
    training_accuracy = model.score(X_train,Y_train)
    print("Training Accuracy: ", training_accuracy)
#=============================================================================
    
    val_vectorizer = CountVectorizer(stop_words="english", max_features=10000, vocabulary=train_vocab)
    X_val = val_vectorizer.fit_transform(val["Body"])
    Y_val = val["Class"]
    Y_val=Y_val.astype('int')
    val_vocab = val_vectorizer.get_feature_names()
    #print(val_vocab==train_vocab)
    
    val_accuracy = model.score(X_val,Y_val)
    print("Validation Accuracy: ", val_accuracy)
    
    new_df = pd.DataFrame()
    new_df["features"] = train_vocab
    
    new_df["weights"] = model.coef_[0]
    ax = sns.distplot(new_df["weights"], kde=False)
    ax.set_yscale('log')
    
    fig= plt.figure(figsize=(12,8))

    sorteddf = new_df.sort_values(by="weights")
    top_10_spam = sorteddf.iloc[0:10]
    top_10_ham = sorteddf.iloc[-10:].iloc[::-1]
    plotguy = pd.concat([top_10_spam,top_10_ham])
    #print(plotguy)
    
    print("Words most associated with conspiracy:",top_10_spam)
    print("Words most associated with tedx:",top_10_ham)
    
    sns.barplot(plotguy["weights"],plotguy["features"])


get_simple_test_set()


