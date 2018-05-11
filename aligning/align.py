# -*- coding: utf-8 -*-
"""
Created on Tue May  8 01:36:35 2018

@author: LukeStacey
"""
import numpy as np
def align(s1,s2,g,m,n):
    NW=np.reshape([0]*(len(s1)+1)*(1+len(s2)),(1+len(s1),1+len(s2)))
    #print(NW)
    out1=[]
    out2=[]
    for i in range(0,len(s1)+1):
        for j in range(0,len(s2)+1):
            if i>0:
                NW[i][j]=NW[i-1][j]+g
            if j>0:
                NW[i][j]=max(NW[i][j],NW[i][j-1]+g)
            if i>0 and j>0:
                if s1[i-1]==s2[j-1]:
                    NW[i][j]=max(NW[i][j],NW[i-1][j-1]+m)
                else:
                    NW[i][j]=max(NW[i][j],NW[i-1][j-1]+n)
    #constructed NW
    #now backtrack
    i=len(s1)
    j=len(s2)
    while i!=0 and j!=0:
        if i>0 and j>0:
            if ( NW[i][j]==NW[i-1][j-1]+m and s1[i-1]==s2[j-1] ) or ( NW[i][j]==NW[i-1][j-1]+n and s1[i-1]!=s2[j-1] ):
                #substitute
                out1.append(s1[i-1])
                out2.append(s2[j-1])
                i=i-1
                j=j-1
                continue
        if i>0:
            if NW[i][j]==NW[i-1][j]+g:
                #insert
                out1.append(s1[i-1])
                out2.append("-")
                i=i-1
                continue
        if j>0:
            if NW[i][j]==NW[i][j-1]+g:
                #delete
                out1.append("-")
                out2.append(s2[j-1])
                j=j-1
                continue
    return out1[::-1],out2[::-1]


def compaligned(s1,s2):
    #s1=strings[0]
    #s2=strings[1]
    score=0
    for i in range(0,len(s1)):
        if s1[i]==s2[i]:
            score=score+1
    if len(s1) != 0:
        return score/len(s1) 
    else:
        return 0

def score(s1,s2):
    score1,score2 =align(s1,s2,0,12,4)
    return compaligned(score1,score2)