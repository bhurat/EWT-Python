# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 10:04:41 2020

@author: bazzz
"""
import numpy as np
from scipy.signal import gaussian

class EWTParams:
    def __init__(self):
        self.log = 0
        self.removeTrends = 'none'
        self.spectrumRegularize = 'none'
        self.lengthFilter = 7
        self.sigmaFilter = 2
        self.typeDetect = 'otsu'
        self.option = 1
        
def spectrumRegularize(f, params):
    if params.spectrumRegularize.lower() == 'gaussian':
        f2 = np.pad(f,params.lengthFilter//2,'reflect')
        Reg_Filter = gaussian(params.lengthFilter,params.sigmaFilter)
        Reg_Filter = Reg_Filter/sum(Reg_Filter)
        f2 = np.convolve(f2,Reg_Filter,mode = 'same')
        return f2[params.lengthFilter//2:-params.lengthFilter//2]
    elif params.spectrumRegularize.lower() == 'average':
        f2 = np.pad(f,params.lengthFilter//2,'reflect')
        Reg_Filter = np.ones(params.lengthFilter)
        Reg_Filter = Reg_Filter/sum(Reg_Filter)
        f2 = np.convolve(f2,Reg_Filter,mode = 'same')
        return f2[params.lengthFilter//2:-params.lengthFilter//2]
    elif params.spectrumRegularize.lower() == 'closing':
        f2 = np.zeros(len(f))
        for i in range(0,len(f)):
            f2[i] = np.min(f[max(0 , i - params.lengthFilter) : min(len(f)-1, i + params.lengthFilter+1)])
        return f2
    
def removeTrends(f, params):
    #still needs to be implemented
    return f