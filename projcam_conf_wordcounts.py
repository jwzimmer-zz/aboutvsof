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

# === === helper dict === === ===

contractions = { 
"ain't": "am not / are not / is not / has not / have not",
"aren't": "are not / am not",
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
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"I'd": "I had / I would",
"I'd've": "I would have",
"I'll": "I shall / I will",
"I'll've": "I shall have / I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
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
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
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
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
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
        self.expand_cont()
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
        return text1
    def expand_cont(self):
        text = self.clean
        text = """scalar weaponry. It’s not just these two guys. Okay? It’s
happening to all of us. You must become aware in order to
resist it, so you need to know that these things, telephone calls,
television programs, various anomalous events that happen in your
midst, pulses sent through stereo equipment. Actually wild
things, we’ve had some things happen up here with some of this
equipment. Same thing. Thank you very much for being here for
this. Thank you so much for coming.  """
        for cont in contractions:
            if "/" in contractions[cont]:
                pass
            else:
                parts = cont.split("'")
                new = ""
                for i in range(1,len(parts)):
                    #print(parts, parts[i-1])
                    new += parts[i-1]+"['|’]"
                new  +=  parts[-1]   
                #print(new)
                recont = re.compile(new, re.IGNORECASE)
                expre = contractions[cont]
                matches = re.findall(recont, text)
                if cont  == "we've":
                    print(matches)
                    print(recont, expre)
                    text = re.sub(recont,expre,text)
                    print(text[-500:])

        #print(text)
        return None
        
    

    

    #lemmas = []
    #wordnet_lemmatizer = WordNetLemmatizer()
        #tokenization = nltk.word_tokenize(body)
        #for w in tokenization:
            #pass
            #lemmas += [wordnet_lemmatizer.lemmatize(w)]


CamelotConfTranscript("ProjectCamelotConference/")


