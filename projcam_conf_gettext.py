#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 14:54:31 2021

@author: jzimmer1
"""

from bs4 import BeautifulSoup
import os
import json

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
    def write_json(self, data, filename):
        with open(filename,"w") as f:
            json.dump(data,f)
        return None

it = CamelotConfTranscript()
rawtext = it.get_transcript_text("ProjectCamelotConference/")
it.write_json(rawtext, "rawtext_pcc_transcripts")
