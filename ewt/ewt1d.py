import numpy as np
from ewt.boundaries import *


def EWT1D(f,params):
    ff = np.fft.fft(f)
    bounds = ewt_boundariesDetect(np.abs(ff[0:int(np.round(len(ff)/2))]),params)
    bounds = bounds*np.pi/(np.round(len(ff)/2))
    
    mfb = EWT_LP_Filterbank(bounds,len(ff))
    
    ewt = np.zeros([len(ff), len(bounds)+1])
    
    for i in range(0,len(bounds)+1):
        ewt[:,i]=np.real(np.fft.ifft(mfb[:,i]*ff));
    return [ewt, mfb, bounds]

def iEWT1D(ewt,mfb):
    rec = np.zeros(len(ewt[:,0]))
    for i in range(0,len(ewt[0,:])):
        rec += np.real(np.fft.ifft(np.fft.fft(ewt[:,i])*mfb[:,i]))
    return rec

def EWT_LP_Filterbank(bounds,N):
    #Calculate Gamma
    gamma = 1;
    for i in range(0,len(bounds)-1):
        r = (bounds[i+1] - bounds[i])/(bounds[i+1]+bounds[i])
        if r < gamma and r > 1e-6:
            gamma = r
    aw = np.arange(0,2*np.pi-1/N,2*np.pi/N)
    aw[np.floor(N/2).astype(int):] -= 2*np.pi 
    aw = np.abs(aw)
    filterbank = np.zeros([N, len(bounds)+1])
    filterbank[:,0] = EWT_LP_Scaling(bounds[0],aw,gamma,N)
    for i in range(1,len(bounds)):
        filterbank[:,i] = EWT_LP_Wavelet(bounds[i-1],bounds[i], aw, gamma, N)
    filterbank[:,len(bounds)] = EWT_LP_Wavelet(bounds[len(bounds)-1],np.pi,aw,gamma,N)
    return filterbank

def EWT_LP_Scaling(w1,aw,gamma,N):
    mbn = (1 - gamma)*w1 #beginning of transition
    pbn = (1 + gamma)*w1 #end of transition
    an = 1/(2*gamma*w1) #scaling in beta function

    yms = 1.0*(aw <= mbn) #if less than lower bound, equals 1
    yms += (aw > mbn)*(aw <= pbn)*np.cos(np.pi*EWT_beta(an*(aw - mbn))/2) #Transition area
    return yms

def EWT_LP_Wavelet(wn,wm,aw,gamma,N):
    if wn > np.pi:  #If greater than pi, subtract 2pi, otherwise dont
        a = 1;
    else:
        a = 0

    mbn = wn - gamma*abs(wn - a*2*np.pi) #beginning of first transition
    pbn = wn + gamma*abs(wn - a*2*np.pi) #end of first transition
    an = 1/(2*gamma*abs(wn - a*2*np.pi)) #scaling in first transition's beta function

    if wm > np.pi: #If greater than pi, subtract 2pi, otherwise dont
        a=1;
    else:
        a=0;
    
    mbm = wm - gamma*abs(wm - a*2*np.pi); #beginning of second transition
    pbm = wm + gamma*abs(wm - a*2*np.pi); #end of second transition
    am = 1/(2*gamma*abs(wm - a*2*np.pi));  #scaling in second transition's beta function
    
    ymw = 1.0*(aw >= pbn)*(aw<= mbm) #equals 1 between transition areas
    ymw += (aw > mbn)*(aw < pbn)*np.sin(np.pi*EWT_beta(an*(aw - mbn))/2) #1st transition area
    if wm < np.pi:
        ymw += (aw > mbm)*(aw < pbm)*np.cos(np.pi*EWT_beta(am*(aw - mbm))/2) #2nd transition area
    else:
        ymw += (aw > mbm)*(aw < pbm)*1.0
    return ymw

def EWT_beta(x):
    bm = (x >= 0)*(x <= 1)*(x**4 * (35 - 84*x + 70*x**2 - 20*x**3))
    bm += (x > 1)
    return bm
