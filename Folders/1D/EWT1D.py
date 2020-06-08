# -*- coding: utf-8 -*-
"""
1D EWT
@author: Basile
"""

def EWT1D(f,params):
    ff = np.fft.fft(f)
    bounds = ewt_boundariesDetect(np.abs(ff[0:int(np.round(ff)/2)]))
    bounds = bounds*np.pi/np.ceil(npround(ff)/2)
    
    mfb = EWT_LP_Filterbank(bounds,len(ff))
    
    ewt = np.zeros([len(ff), len(bounds)+1])
    
    for i in range(0,len(bounds)+1):
        ewt[:,i]=np.real(np.fft.ifft(mfb[:,i]*ff));
    return [ewt, mfb, bounds]