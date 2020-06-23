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



Bound = 0 #show bounds (NOT IMPLEMENTED)
Comp = 0 #show components
Rec = 1 #show reconstruction 

params = EWTParams()
params.log = 1
transform = 'curvelet'

f = np.genfromtxt('Tests/2d/building.csv', delimiter=',')
f = f[0:-1,0:-3]
plt.imshow(f,cmap = 'gray')
plt.show()
f = (f-np.min(f))/(np.max(f)-np.min(f))


if transform.lower() == 'tensor':
    [ewtc, mfb_row, mfb_col,bounds_row, bounds_col] = ewt2dTensor(f,params)
elif transform.lower() == 'curvelet':
    [ewtc, mfb, bounds_scales, bounds_angles] = ewt2dCurvelet(f,params)

if Comp == 1:
    for i in range(0,len(mfb_row[0,:])):
        for j in range(0,len(mfb_col[0,:])):
            plt.imshow(ewtc[i][j],cmap = 'gray')
            plt.show()
if Rec == 1:
    if transform.lower() == 'tensor':
        recon = iewt2dTensor(ewtc, mfb_row, mfb_col)
    elif transform.lower() == 'curvelet':
        recon = iewt2dCurvelet(ewtc, mfb)
    plt.imshow(recon,cmap = 'gray')
    plt.show()