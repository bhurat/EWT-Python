# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:09:30 2020

@author: bazzz
"""

import numpy as np
from scipy.special import iv
import matplotlib.pyplot as plt
from ewt.ewt2d import *
from ewt.utilities import EWTParams

params = EWTParams()


Bound = 0
Comp = 0
Rec = 1
TFplane = 0


f = np.genfromtxt('Tests/texture.csv', delimiter=',')
f = f[0:-1,0:-3]
plt.imshow(f,cmap = 'gray')
plt.show()
f = (f-np.min(f))/(np.max(f)-np.min(f))

[ewtc, mfb_row, mfb_col,bounds_row, bounds_col] = ewt2dTensor(f,params)

if Comp == 1:
    for i in range(0,len(mfb_row[0,:])):
        for j in range(0,len(mfb_col[0,:])):
            plt.imshow(ewtc[i][j],cmap = 'gray')
            plt.show()

if Rec == 1:
    recon = iewt2dTensor(ewtc, mfb_row,mfb_col)
    plt.imshow(recon,cmap = 'gray')
    plt.show()