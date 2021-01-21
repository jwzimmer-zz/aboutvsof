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

def make_filemap():
    dicti = {
        "‘Depraved’_ Rep. Jason Crow condemns Rep. Marjorie Taylor Greene’s rhetoric ahead of impeachment vote - POLITICO.html":0,
        "‘He knows he’s screwing with everyone’_ Gosar embraces conspiracy theories.htm":0,
        "An overview of the “Out of Shadows” documentary – Est. 1933.html":1,
        "Apple, Twitter, Google, and Instagram Collude to Defeat Trump _ Observer.htm":1,
        "Creative Coalition Rejects $10k Donation From Veterans—Because ‘Trump’ _ Observer.htm":1,
        "Does the Jeffrey Epstein indictment prove Qanon right_.htm":1,
        "Donald Trump Laura Ingraham Interview Transcript August 31_ Says People _in the Dark Shadows_ Controlling Biden - Rev.htm":1,
        "Exclusive_ Hillary Clinton Campaign Systematically Overcharging Poorest Donors _ _ Observer.htm":1,
        "Facebook and YouTube Show Frantic Alliegence to Clinton _ Observer.htm":1,
        "Fact check_ 35,000 “malnourished” and “caged” children were not recently rescued from tunnels by U.S. military _ Reuters.html":0,
        "House votes to condemn baseless QAnon conspiracy theory - The Washington Post.htm":0,
        "How QAnon Became Obsessed With ‘Adrenochrome,’ an Imaginary Drug Hollywood Is Harvesting from Kids.htm":0,
        "Inside ‘Out of Shadows,’ the Bonkers Hollywood-Pedophilia Documentary QAnon Loves.htm":0,
        "Is YouTube Preventing People From Finding The Out Of Shadows Documentary_ _ Small Screen.htm":1,
        "Media Favoritism of Clinton Extends Beyond News Programs _ _ Observer.htm":1,
        "Media Orgs Donate to Clinton Foundation Then Downplay Clinton Foundation Scandal _ _ Observer.htm":1,
        "Milo Yiannopoulos Advises Donald Trump to Quit Twitter _ Observer.htm":1,
        "National Enquirer Never Would Have Run BuzzFeed’s Trump Dossier _ Observer.htm":1,
        "proqexcerpts.txt":1,
        "QAnon and Child Trafficking_ Is 'Save the Children' a Conspiracy_.htm":0,
        "QAnon Is More Important Than You Think - The Atlantic.htm":0,
        "QAnon Promotes Pedo-Ring Conspiracy Theories. Now They’re Stealing Kids..html":0,
        "QAnon_ the religion of conspiracy - Elcano Blog.htm":0,
        "Shia LaBeouf’s Behavior at Trump Protest Is Hypocrisy at Its Finest _ Observer.htm":1,
        "Team Trump Isn’t Hiding Support for QAnon Kooks Anymore After Marjorie Taylor Greene’s Win in Georgia Primary.htm":0,
        "The Dark Virality of a Hollywood Blood-Harvesting Conspiracy _ WIRED.htm":0,
        "Timothy Holmseth - Guest Profile.htm":1,
        "Timothy Holmseth Update_ Deep State Harassment, Jury Trial _ Listen Notes.htm":1,
        "Viral Chart Distorts Human Trafficking Statistics - FactCheck.org.html":0
        }
    write_json(dicti, "q_filemap.json")
    return None

make_filemap()

class Text():
    def __init__(self):
        pass
        
    def get_text_from_website(self,filename):
        soup = BeautifulSoup(open(self.DirName+filename,encoding="ISO-8859-1"),features="lxml")
        allps = soup.find_all('p')
        alltext = []
        for p in allps:
            alltext += [p.get_text()]
        return alltext
    
    