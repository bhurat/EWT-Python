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
show_orig = 1   #show original signal
show_bounds = 1 #show bounds 
show_coefs = 1  #show components
show_recon = 1  #show reconstruction 

f = np.genfromtxt('Tests/1d/sig2.csv', delimiter=',')

params = ewt_params()

[ewtc, mfb, bounds] = ewt1d(f,params)

if show_orig == 1:
    plt.figure()
    plt.suptitle('Original signal')
    plt.plot(f)
    plt.show()
if show_bounds == 1:
    showewt1dBoundaries(f,bounds)
if show_coefs == 1:
    showEWT1DCoefficients(ewtc)
if show_recon == 1:
    recon = iewt1d(ewtc,mfb)
    plt.figure()
    plt.plot(recon)
    plt.suptitle('Reconstructed signal')
    plt.show()
    print(f'Reconstruction difference: {sum((recon -f)**2)}')
