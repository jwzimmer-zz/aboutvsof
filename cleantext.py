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

class QModel():
    def __init__(self):
        self.df = self.get_df()
        self.run_model(self.df)
        
    def get_text_from_website(self,filename):
        soup = BeautifulSoup(open(self.DirName+filename,encoding="ISO-8859-1"),features="lxml")
        allps = soup.find_all('p')
        alltext = []
        for p in allps:
            alltext += [p.get_text()]
        return alltext

    def get_df(self,tokenlist0,tokenlist1,n=50):
        #tokenlists should be json list objects (from a file) or python list objects
        #0 for non-con, 1 for con; n is the chunk size (number of words in a data point)
        #expected format = df has Body (text) and Class columns (0/1)
        #df has all the training data in it
        df = pd.DataFrame(columns=["Body","Class"])
        newdf = []
        chunks0 = [tokenlist0[i:i + n] for i in range(0, len(tokenlist0), n)]
        chunks1 = [tokenlist1[i:i + n] for i in range(0, len(tokenlist1), n)]
        for t in chunks0:
            #print(t)
            strt = ""
            for i in t:
                strt += " "+i
            newdf += [{"Body":strt,"Class":0}]
        for p in chunks1:
            strp = ""
            for i in p:
                strp += " "+i
            newdf += [{"Body":strp,"Class":1}]
        df = df.append(pd.DataFrame(newdf),ignore_index=True)
        #print(df.head())
        return df
        
    def run_model(self,df):
    
        train, val = train_test_split(df, test_size=0.2, random_state=42)
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

        sns.barplot(plotguy["weights"],plotguy["features"])
        

QModel()