# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:22:37 2020

@author: bazzz
"""

def iEWT1D(ewt,mfb):
    rec = np.zeros(len(ewt[:,0]))
    for i in range(0,len(ewt[0,:])):
        rec += np.real(np.fft.ifft(np.fft.fft(ewt[:,i])*mfb[:,i]))
    return rec