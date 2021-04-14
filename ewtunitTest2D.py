# -*- coding: utf-8 -*-

import unittest
from ewt import ewt2d
import ewt
import numpy as np

class Testewt2dMethods(unittest.TestCase):
    def testewt2d_curveletWavelet1(self):
        #Tests wavelet construction for curvelets (polar wedge)
        h = 127; w = 127;
        radii = np.zeros([h,w])
        theta = np.zeros([h,w])
        h_center = h//2 + 1; w_center = w//2+1
        for i in range(0,h):
            for j in range(0,w):
                ri = (i+1.0 - h_center)*np.pi/h_center
                rj = (j+1.0 - w_center)*np.pi/w_center
                radii[i,j] = np.sqrt(ri**2 + rj**2)
                theta[i,j] = np.arctan2(ri,rj)
        wav = ewt2d.ewt2d_curveletWavelet(theta,radii,.3,.5,.3,.7,.05,.05)
        self.assertTrue(np.max(wav) == 1)
    def testewt2d_curveletWavelet2(self):
        #Tests wavelet construction for curvelets if only one angle detected (polar wedge)
        h = 127; w = 127;
        radii = np.zeros([h,w])
        theta = np.zeros([h,w])
        h_center = h//2 + 1; w_center = w//2+1
        for i in range(0,h):
            for j in range(0,w):
                ri = (i+1.0 - h_center)*np.pi/h_center
                rj = (j+1.0 - w_center)*np.pi/w_center
                radii[i,j] = np.sqrt(ri**2 + rj**2)
                theta[i,j] = np.arctan2(ri,rj)
        wav = ewt2d.ewt2d_curveletWavelet(theta,radii,.3,.3+np.pi,.3,.7,.1,.1)
        self.assertTrue(np.max(wav) == 1)
    def testewt2d_curveletWavelet3(self):
        #Tests wavelet construction for curvelets if one angle detected + epsilon(polar wedge)
        h = 127; w = 127;
        radii = np.zeros([h,w])
        theta = np.zeros([h,w])
        h_center = h//2 + 1; w_center = w//2+1
        for i in range(0,h):
            for j in range(0,w):
                ri = (i+1.0 - h_center)*np.pi/h_center
                rj = (j+1.0 - w_center)*np.pi/w_center
                radii[i,j] = np.sqrt(ri**2 + rj**2)
                theta[i,j] = np.arctan2(ri,rj)
        wav = ewt2d.ewt2d_curveletWavelet(theta,radii,.3,.3+np.pi-.0000001,.3,.7,.1,.1)
        self.assertTrue(np.max(wav) == 1)
    def testewt2dCurvelet1Even(self):
        #Tests empirical curvelet transform + reconstruction for even images
        f = np.genfromtxt('Tests/2d/texture.csv', delimiter=',')
        f = (f-np.min(f))/(np.max(f)-np.min(f))
        params = ewt.utilities.ewt_params()
        [ewtc,mfb,b1,b2] = ewt2d.ewt2dCurvelet(f, params)
        recon = ewt2d.iewt2dCurvelet(ewtc,mfb)
        self.assertTrue(np.sum((recon-f)**2) < 10**(-10))
    def testewt2dCurvelet1Odd(self):
        #Tests empirical curvelet transform + reconstruction for odd images
        f = np.genfromtxt('Tests/2d/texture.csv', delimiter=',')
        f = (f-np.min(f))/(np.max(f)-np.min(f))
        f = f[0:-1,0:-1]
        params = ewt.utilities.ewt_params()
        [ewtc,mfb,b1,b2] = ewt2d.ewt2dCurvelet(f, params)
        recon = ewt2d.iewt2dCurvelet(ewtc,mfb)
        self.assertTrue(np.sum((recon-f)**2) < 10**(-10))
        
if __name__ == '__main__':
    unittest.main()