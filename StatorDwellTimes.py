#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:00:27 2020

@author: vaulthigh
"""


import numpy as np
import matplotlib.pyplot as plt
from statorBinding_model_2 import KMC3states_maketrace


def PlotDwellTimes(model='TwoStateCatchSimple',kuw=.01,kwu=.01,kws=.01,ksw=.01,kus=0,ksu=0,N0=0):
    
    #Split N0 between w and s, and run sim long enough so intial conditions don't matter
    Nw0=int(N0/2)
    Ns0=N0-Nw0
    
    if model in ['TwoStateCatchSimple','Yuan2019','Intermediate']:
        kus,ksu = 0,0
        
    time,Ns,Nw =KMC3states_maketrace(kuw=kuw,kwu=kwu,kws=kws,ksw=ksw,kus=kus,ksu=ksu,Ns0=Ns0,Nw0=Nw0,Npts=1e4)
    if model in ['TwoStateCatchSimple','TwoStateCatchTrad']:
        N = Ns+Nw
    elif model in ['Yuan2019']:   
        N = Nw
    elif model in ['Intermediate','TriangleIntermediate']:   
        N = Ns
    dN = np.diff(N)
    i = np.where(np.abs(dN)>0)
    ti = time[i]
    dwells = np.diff(ti)
    
    plt.figure('StatDwellTimes')
    plt.hist(dwells, 200,histtype='step')
    plt.xlabel('time (s)');plt.ylabel('counts');plt.yscale('log')
    
    return