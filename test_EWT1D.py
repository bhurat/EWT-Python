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
from ewt.tests import *
from ewt.utilities import ewt_params

plt.close('all')
show_bounds = 0 #show bounds 
show_coefs = 0 #show components
show_recon = 1 #show reconstruction 


params = ewt_params()

f = np.genfromtxt('Tests/1d/sig2.csv', delimiter=',')

[ewt, mfb, bounds] = ewt1d(f,params)

if show_coefs == 1:
    for i in range(0,len(ewt[0])):
        plt.plot(ewt[i])
        plt.show()
if show_recon == 1:
    recon = iewt1d(ewt,mfb)
    plt.plot(recon)
    print(f'Reconstruction difference: {sum((recon -f)**2)}')
if show_bounds == 1:
    showewt1dBoundaries(f,bounds)