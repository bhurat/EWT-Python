import numpy as np
from ewt.boundaries import *

"""
ewt1d(f,params)
Performs the 1D empirical wavelet transform
Input:
    f       - 1D signal
    params  - parameters for EWT (see utilities)
Output:
    ewt     - empirical wavelet coefficients
    mfb     - empirical wavelet filter bank
    bounds  - bounds detected on Fourier spectrum
Author: Basile Hurat, Jerome Gilles"""

def ewt1d(f,params):
    ff = np.fft.fft(f) 
    
    #performs boundary detection
    bounds = ewt_boundariesDetect(np.abs(ff[0:int(np.round(len(ff)/2))]),params)
    bounds = ewt_boundariesDetect(np.abs(ff),params)
    print(bounds)
    bounds = bounds*np.pi/(np.round(len(ff)/2))
    
    #From bounds, construct filter bank
    mfb = ewt_LP_Filterbank(bounds,len(ff))
    
    #Filter to get empirical wavelet coefficients
    ewt = []
    for i in range(0,len(bounds)+1):
        ewt.append(np.real(np.fft.ifft(mfb[i]*ff)));
    return [ewt, mfb, bounds]

"""
iewt1d(f,params)
Performs the inverse 1D empirical wavelet transform
Input:
    ewt     - empirical wavelet coefficients
    mfb     - empirical wavelet filter bank
Output:
    rec     - reconstruction of signal which generated empirical wavelet 
          coefficients and empirical wavelet filter bank
Author: Basile Hurat, Jerome Gilles"""

def iewt1d(ewt,mfb):
    rec = np.zeros(len(ewt[0]))
    for i in range(0,len(ewt)):
        rec += np.real(np.fft.ifft(np.fft.fft(ewt[i])*mfb[i]))
    return rec

"""
ewt_LP_Filterbank(bounds,N)
Construct empirical wavelet filterbank based on a set of boundaries in the 
Fourier spectrum
Input:
    bounds  - detected bounds in the Fourier spectrum
    N       - desired size of filters (usually size of original signal)
Output:
    mfb     - resulting empirical wavelet filter bank
Author: Basile Hurat, Jerome Gilles"""
def ewt_LP_Filterbank(bounds,N):
    #Calculate Gamma
    gamma = 1;
    for i in range(0,len(bounds)-1):
        r = (bounds[i+1] - bounds[i])/(bounds[i+1]+bounds[i])
        if r < gamma and r > 1e-16:
            gamma = r
    gamma *= (1-1/N) #ensures strict inequality
    
    
    aw = np.arange(0,2*np.pi-1/N,2*np.pi/N)
    aw[np.floor(N/2).astype(int):] -= 2*np.pi 
    aw = np.abs(aw)
    filterbank = []
    filterbank.append(ewt_LP_Scaling(bounds[0],aw,gamma,N))
    for i in range(1,len(bounds)):
        filterbank.append(ewt_LP_Wavelet(bounds[i-1],bounds[i], aw, gamma, N))
    filterbank.append(ewt_LP_Wavelet(bounds[-1],np.pi,aw,gamma,N))
    return filterbank

"""
ewt_LP_Scaling(w1,aw,gamma,N)
Constructs empirical scaling function, which is the low-pass filter for EWT 
filterbank
Input:
    w1      - first bound, which delineates the end of low-pass filter
    aw      - reference vector which goes from 0 to 2pi
    gamma   - gamma value which guarantees tight frame
Output:
    yms     - resulting empirical scaling function
Author: Basile Hurat, Jerome Gilles"""
def ewt_LP_Scaling(w1,aw,gamma,N):
    mbn = (1 - gamma)*w1 #beginning of transition
    pbn = (1 + gamma)*w1 #end of transition
    an = 1/(2*gamma*w1) #scaling in beta function

    yms = 1.0*(aw <= mbn) #if less than lower bound, equals 1
    yms += (aw > mbn)*(aw <= pbn)*np.cos(np.pi*ewt_beta(an*(aw - mbn))/2) #Transition area
    return yms

"""
ewt_LP_Wavelet(wn, wm, aw, gamma, N)
Constructs empirical wavelet, which is a band-pass filter for EWT filterbank
Input:
    wn      - lower bound, which delineates the beginning of band-pass filter
    wm      - higher bound, which delineates the end of band-pass filter
    aw      - reference vector which goes from 0 to 2pi
    gamma   - gamma value which guarantees tight frame
Output:
    ymw     - resulting empirical wavelet 
Author: Basile Hurat, Jerome Gilles"""
def ewt_LP_Wavelet(wn,wm,aw,gamma,N):
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
    
    mbm = wm - gamma*abs(wm - a*2*np.pi) #beginning of second transition
    pbm = wm + gamma*abs(wm - a*2*np.pi) #end of second transition
    am = 1/(2*gamma*abs(wm - a*2*np.pi))  #scaling in second transition's beta function
    
    ymw = 1.0*(aw > mbn)*(aw< pbm) #equals 1 between transition areas
    case = (aw > mbn)*(aw < pbn)
    ymw[case] *= np.sin(np.pi*ewt_beta(an*(aw[case] - mbn))/2) #1st transition area
    if wm < np.pi:
        case = (aw > mbm)*(aw < pbm)
        ymw[case] *= np.cos(np.pi*ewt_beta(am*(aw[case] - mbm))/2) #2nd transition area
        
    return ymw

"""
ewt_beta(x)
Beta function that is used in empirical wavelet and empirical scaling function
construction
Input:
    x       - vector x as an input
Output:
    bm      - beta function applied to vector
Author: Basile Hurat, Jerome Gilles"""
def ewt_beta(x):
    bm = (x >= 0)*(x <= 1)*(x**4 * (35 - 84*x + 70*x**2 - 20*x**3))
    bm += (x > 1)
    return bm
