# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:44:06 2020
Generates the results given by the 1D empirical wavelet transforms, which 
are described in the papers
J. Gilles, "Empirical Wavelet Transform", IEEE Trans. on Signal Processing, 
2013

Feel free to try with your own signals and change parameters. 
@author: Basile Hurat
"""

import numpy as np
from scipy.special import iv
import matplotlib.pyplot as plt
from ewt.ewt1d import *
from ewt.utilities import ewt_params

plt.close('all')
show_orig = 0   #show original signal
show_bounds = 0 #show bounds 
show_coefs = 0  #show components
show_recon = 0  #show reconstruction 
signal = 'csig1'

real = 1;
if signal.lower() == 'csig1':
    fR = np.genfromtxt('Tests/1d/csig1R.csv', delimiter=',')
    fI = np.genfromtxt('Tests/1d/csig1I.csv', delimiter=',')
    f = fR + fI*1j
    real = 0
else: 
    f = np.genfromtxt('Tests/1d/'+signal+'.csv', delimiter=',')
params = ewt_params()

#Detect options include: 
#Scalespace, locmax, locmaxmin, locmaxminf, adaptive, adaptivreg
params.detect = 'adaptive'
params.typeDetect = 'otsu'
[ewtc, mfb, bounds] = ewt1d(f,params)

if show_orig == 1:
    plt.figure()
    plt.suptitle('Original signal')
    plt.plot(np.abs(f))
    plt.show()
if show_bounds == 1:
    print(f'# of detected bounds: {len(bounds)}')
    showewt1dBoundaries(f,bounds)
if show_coefs == 1:
    showEWT1DCoefficients(ewtc)
if show_recon == 1:
    recon = iewt1d(ewtc,mfb)
    plt.figure()
    plt.plot(np.abs(recon))
    plt.suptitle('Reconstructed signal')
    plt.show()
    if real:
        print(f'Reconstruction difference: {np.linalg.norm(recon -f)}')
    else:
        print(f'Reconstruction difference: {np.linalg.norm(recon-f)}')