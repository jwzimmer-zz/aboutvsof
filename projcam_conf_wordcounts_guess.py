#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 14:54:31 2021

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

import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
from nltk.stem import 	WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

# === === helper dict === === ===

contractions = { 
"ain't": "are not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I had",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "sshe will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there had",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they had",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

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
    def __init__(self, DirName):
        self.DirName = DirName
        self.text = self.get_transcript_text()
        self.clean = self.clean_text()
        self.exp = self.expand_cont()
        self.wordlist_snowball()
    def get_transcript_text(self):
        allfiles = os.listdir(self.DirName)
        alltext = ""
        for f in allfiles:
            soup = BeautifulSoup(open(self.DirName+f,encoding="ISO-8859-1"),features="lxml")
            allps = soup.find_all('p',class_="style7")
            for p in allps:
                alltext += p.get_text()
        return alltext
    def clean_text(self):
        #remove everything between [...]
        asidespattern = r'\[.*\]'
        asidesmatches = re.findall(asidespattern, self.text)
        new_text = re.sub(asidespattern,'',self.text)
        #remove the transcription team note between **...**
        notepattern = r'\*\*.*\*\*'
        notematches = re.findall(notepattern, new_text)
        text = re.sub(notepattern,'',new_text)
        #remove the text "Click here for the video interview"
        text1 = re.sub(r'Click here for the video interview','',text)
        #remove people's inititals (indicating speaker)
        text2 = re.sub(r'DO[\'|’]F[\W]*[[0-9][0-9]]*:|DO[\'|’]F:','',text1)
        text3 = re.sub(r'[A-Z][A-Z][\W]*[[0-9][0-9]]*:|[A-Z][A-Z]:','',text2)
        #matches = re.findall(r'[A-Z][A-Z][\W]*[[0-9][0-9]]*:|[A-Z][A-Z]:',text2)
        #print(matches)
        #print(text3)
        return text3
    def expand_cont(self):
        text = self.clean
        for cont in contractions:
            if "/" in contractions[cont]:
                pass
            else:
                parts = cont.split("'")
                new = ""
                for i in range(1,len(parts)):
                    new += parts[i-1]+"['|’]"
                new  +=  parts[-1]   
                recont = re.compile(new, re.IGNORECASE)
                expre = contractions[cont]
                matches = re.findall(recont, text)
                text = re.sub(recont,expre,text)
        #write_json(text, "guess_ambiguous_contractions.json")
        return text
    def wordlist_snowball(self):
        text = self.exp.lower()
        lemmas = []
        wordnet_lemmatizer = WordNetLemmatizer()
        tokenization = nltk.word_tokenize(text)
        
        stemmer = SnowballStemmer("english")
        #print(tokenization)
        for w in tokenization:
            #pass
            #lemmas += [wordnet_lemmatizer.lemmatize(w)]
            lemmas += [stemmer.stem(w)]
        #print(lemmas)
        lemmadict = {}
        for w in lemmas:
            if w in lemmadict:
                lemmadict[w] += 1
            else:
                lemmadict[w] = 1
        #print(lemmadict)
        write_json(lemmadict, "snowball_stem_PCC_freq.json")
        return None

        
    

    

    

CamelotConfTranscript("ProjectCamelotConference/")


