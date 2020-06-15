# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 13:59:13 2020

@author: bazzz
"""
import numpy as np
from ewt.boundaries import *
from ewt.ewt1d import *
def ewt2dTensor(f,params):
    ff = np.fft.fft(f.T) #take 1D fft along columns
    meanfft = np.mean(np.abs(ff),axis = 0) #take average of magnitude
    bounds_col = ewt_boundariesDetect(meanfft[0:int(np.round(len(meanfft)/2))],params)
    bounds_col = bounds_col*2*np.pi/len(meanfft)
    ewtc_col = []
    mfb_col = EWT_LP_Filterbank(bounds_col,len(meanfft)) #construct filter bank
    for i in range(0,len(mfb_col[0,:])):
        filter_col = np.tile(mfb_col[:,i],[len(f[0,:]),1]) #repeat down
        ewtc_col.append(np.real(np.fft.ifft(filter_col*ff)).T) #get coefficients,
    
    # REPEAT WITH ROWS #
    ff = np.fft.fft(f) #take 1D fft along rows
    meanfft = np.mean(np.abs(ff),axis = 0) 
    bounds_row = ewt_boundariesDetect(meanfft[0:int(np.round(len(meanfft)/2))],params)
    bounds_row = bounds_row*2*np.pi/len(meanfft)

    ewtc = []
    mfb_row = EWT_LP_Filterbank(bounds_row,len(meanfft))
    for i in range(0,len(mfb_row[0,:])):
        ewtc_row = []
        filter_row = np.tile(mfb_row[:,i],[len(f[:,0]),1])
        for j in range(0,len(mfb_col[0,:])):
            ff = np.fft.fft(ewtc_col[j])
            ewtc_row.append(np.real(np.fft.ifft(filter_row*ff)))
        ewtc.append(ewtc_row)
            
    return [ewtc, mfb_row, mfb_col,bounds_row,bounds_col]
    
def iewt2dTensor(ewtc,mfb_row,mfb_col):
    [h,w]= ewtc[0][0].shape
    ewt_col = []
    for i in range(0,len(mfb_col[0,:])):
        for j in range(0,len(mfb_row[0,:])):
            ff = np.fft.fft(ewtc[j][i])
            filter_row = np.tile(mfb_row[:,j],[h, 1])
            if j == 0:
                ewt_col.append(np.zeros([h,w]))
            ewt_col[i] += np.real(np.fft.ifft(filter_row*ff))
    
    for i in range(0,len(mfb_col[0,:])):
        ff = np.fft.fft(ewt_col[i].T);
        if i == 0:
            img = np.zeros([h,w])
        
        filter_col = np.tile(mfb_col[:,i],[w, 1])
        img = img + np.real(np.fft.ifft(filter_col*ff)).T
    return img
        
