# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:44:06 2020

@author: bazzz
"""

import numpy as np
from scipy.special import iv
import matplotlib.pyplot as plt
from ewt.ewt1d import *
from ewt.tests import *
from ewt.utilities import EWTParams

params = EWTParams()

f = np.genfromtxt('Tests/1d/seismic.csv', delimiter=',')

[ewt, mfb, bounds] = EWT1D(f,params)

show_coefs = 0
show_recon = 1

if show_coefs == 1:
    for i in range(0,len(ewt[0,:])):
        plt.plot(ewt[:,i])
        plt.show()
if show_recon == 1:
    recon = iEWT1D(ewt,mfb)
    plt.plot(recon)