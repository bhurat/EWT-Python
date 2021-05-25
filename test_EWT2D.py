# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:09:30 2020
Generates the results given by various 2D empirical wavelet transforms, which 
are described in the papers
J. Gilles, "Empirical Wavelet Transform", IEEE Trans. on Signal Processing, 
2013
J. Gilles, G. Tran, S. Osher, "2D Empirical tranforms. Wavelets, Ridgelets and
Curvelets Revisited" submitted at SIAM Journal on Imaging Sciences. 2013 

Feel free to try with your own images and change parameters. 
@author: Basile Hurat
"""

import numpy as np
from scipy.special import iv
import matplotlib.pyplot as plt
from ewt.ewt2d import *
from ewt.utilities import ewt_params

plt.close('all')

show_orig = 1   #show original image
show_bounds = 0 #show bounds 
show_coefs = 0  #show components
show_recon = 1  #show reconstruction 


params = ewt_params()

transform = 'curvelet'
params.typeDetect = 'otsu'
f = np.genfromtxt('Tests/2d/texture.csv', delimiter=',')

#f = (f-np.min(f))/(np.max(f)-np.min(f))
#f = f[0:-1,0:-1]

if transform.lower() == 'tensor':
    [ewtc, mfb_row, mfb_col,bounds_row, bounds_col] = ewt2dTensor(f,params)
elif transform.lower() == 'lp':
    [ewtc, mfb, bounds_scales] = ewt2dLP(f,params)
elif transform.lower() == 'ridgelet':
    [ewtc, mfb, bounds_scales] = ewt2dRidgelet(f,params)
elif transform.lower() == 'curvelet':
    [ewtc, mfb, bounds_scales, bounds_angles] = ewt2dCurvelet(f,params)

if show_orig == 1:
    plt.imshow(f,cmap = 'gray')
    plt.suptitle('Original image')
    plt.show()

if show_bounds == 1:
    if transform.lower() == 'tensor':
        show2DTensorBoundaries(f,bounds_row,bounds_col)
    elif transform.lower() == 'lp':
        show2DLPBoundaries(f,bounds_scales)
    elif transform.lower() == 'ridgelet':
        show2DLPBoundaries(f,bounds_scales)
    elif transform.lower() == 'curvelet':
        show2DCurveletBoundaries(f,params.option,bounds_scales,bounds_angles)

if show_coefs == 1:
    showEWT2DCoefficients(ewtc, transform)

if show_recon == 1:
    if transform.lower() == 'tensor':
        recon = iewt2dTensor(ewtc, mfb_row, mfb_col)
    elif transform.lower() == 'lp':
        recon = iewt2dLP(ewtc,mfb)
    elif transform.lower() == 'ridgelet':
        recon = iewt2dRidgelet(ewtc,mfb)
    elif transform.lower() == 'curvelet':
        recon = iewt2dCurvelet(ewtc, mfb)
    print(f'recon difference: {np.sum((recon - f)**2)}')
    plt.figure()
    plt.suptitle('Reconstructed image')
    plt.imshow(recon,cmap = 'gray')
    plt.show()
