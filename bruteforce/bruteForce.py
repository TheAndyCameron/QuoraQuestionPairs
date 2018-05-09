# -*- coding: utf-8 -*-
"""
Created on Tue May  8 00:45:54 2018

@author: LukeStacey
"""


def bruteforce(Q1s,Q2s):
    match=[]
    for i in range(0,len(Q1s)):
        if len(Q1s[i])>len(Q2s[i]):
            s1=Q1s[i]
            s2=Q2s[i]
        else:
            s1=Q2s[i]
            s2=Q1s[i]
        matches=0
        for c in s1:
            place=s2.find(c)
            if place !=-1:
                matches=matches+1                
                s2=s2[:place]+s2[place+1:]                
        match.append(100*matches/len(s1))
    return match

#Brute force
import numpy as np
Q1s=["Cheese", "Bread", "Feta","Mead","wardrobe","haloumi"]
Q1s=np.array(Q1s)
Q2s=["Chair", "table","Fouton", "bunk Bed", "wardrobe","chez longe"]
Q2s=np.array(Q2s)
similarity=bruteforce(Q1s,Q2s)
print(similarity)

