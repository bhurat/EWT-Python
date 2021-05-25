# -*- coding: utf-8 -*-

import unittest
from ewt import ewt2d
from ewt import boundaries
from ewt import ewt1d

import ewt
import numpy as np

class EWTBlackboxTest(unittest.TestCase):
    def setUp(self):
        self.f = np.genfromtxt('Tests/2d/texture.csv', delimiter=',')
        self.f = (self.f-np.min(self.f))/(np.max(self.f)-np.min(self.f))
        self.params = ewt.utilities.ewt_params()
    def test_ewt1dEven(self):
        f = np.genfromtxt('Tests/1d/sig2.csv', delimiter=',')
        [ewtc,mfb,bounds] = ewt1d.ewt1d(f,self.params)
        recon = ewt1d.iewt1d(ewtc,mfb)
        self.assertTrue(np.sum((recon-f)**2) < 10**(-10))
    def test_ewt1dOdd(self):
        f = np.genfromtxt('Tests/1d/sig2.csv', delimiter=',')
        f = f[0:-1]
        [ewtc,mfb,bounds] = ewt1d.ewt1d(f,self.params)
        recon = ewt1d.iewt1d(ewtc,mfb)
        self.assertTrue(np.sum((recon-f)**2) < 10**(-10))
    def testewt2dCurvelet1Even(self):
        #Tests empirical curvelet transform + reconstruction for even images
        [ewtc,mfb,b1,b2] = ewt2d.ewt2dCurvelet(self.f, self.params)
        recon = ewt2d.iewt2dCurvelet(ewtc,mfb)
        self.assertTrue(np.sum((recon-self.f)**2) < 10**(-10))
    def testewt2dCurvelet1Odd(self):
        #Tests empirical curvelet transform + reconstruction for odd images
        self.f = self.f[0:-1,0:-1]
        [ewtc,mfb,b1,b2] = ewt2d.ewt2dCurvelet(self.f, self.params)
        recon = ewt2d.iewt2dCurvelet(ewtc,mfb)
        self.assertTrue(np.sum((recon-self.f)**2) < 10**(-10))
    def testewt2dCurvelet2Even(self):
        #Tests empirical curvelet transform + reconstruction for even images
        self.params.option = 2
        [ewtc,mfb,b1,b2] = ewt2d.ewt2dCurvelet(self.f, self.params)
        recon = ewt2d.iewt2dCurvelet(ewtc,mfb)
        self.assertTrue(np.sum((recon-self.f)**2) < 10**(-10))
    def testewt2dCurvelet2Odd(self):
        #Tests empirical curvelet transform + reconstruction for odd images
        self.f = self.f[0:-1,0:-1]
        self.params.option = 2
        [ewtc,mfb,b1,b2] = ewt2d.ewt2dCurvelet(self.f, self.params)
        recon = ewt2d.iewt2dCurvelet(ewtc,mfb)
        self.assertTrue(np.sum((recon-self.f)**2) < 10**(-10))
    def testewt2dCurvelet3Even(self):
        #Tests empirical curvelet transform + reconstruction for even images
        self.params.option = 3
        [ewtc,mfb,b1,b2] = ewt2d.ewt2dCurvelet(self.f, self.params)
        recon = ewt2d.iewt2dCurvelet(ewtc,mfb)
        self.assertTrue(np.sum((recon-self.f)**2) < 10**(-10))
    def testewt2dCurvelet3Odd(self):
        #Tests empirical curvelet transform + reconstruction for odd images
        self.f = self.f[0:-1,0:-1]
        self.params.option = 3
        [ewtc,mfb,b1,b2] = ewt2d.ewt2dCurvelet(self.f, self.params)
        recon = ewt2d.iewt2dCurvelet(ewtc,mfb)
        self.assertTrue(np.sum((recon-self.f)**2) < 10**(-10))
    def testewt2dTensorEven(self):
        #Tests empirical tensor transform + reconstruction for odd images
        [ewtc,mfb1,mfb2,b1,b2] = ewt2d.ewt2dTensor(self.f, self.params)
        recon = ewt2d.iewt2dTensor(ewtc,mfb1,mfb2)
        self.assertTrue(np.sum((recon-self.f)**2) < 10**(-10))
    def testewt2dTensorOdd(self):
        #Tests empirical tensor transform + reconstruction for odd images
        self.f = self.f[0:-1,0:-1]
        [ewtc,mfb1,mfb2,b1,b2] = ewt2d.ewt2dTensor(self.f, self.params)
        recon = ewt2d.iewt2dTensor(ewtc,mfb1,mfb2)
        self.assertTrue(np.sum((recon-self.f)**2) < 10**(-10))
    def testewt2dLPEven(self):
        #Tests empirical LP transform + reconstruction for odd images
        [ewtc,mfb,bounds] = ewt2d.ewt2dLP(self.f, self.params)
        recon = ewt2d.iewt2dLP(ewtc,mfb)
        self.assertTrue(np.sum((recon-self.f)**2) < 10**(-10))
    def testewt2dLPOdd(self):
        #Tests empirical LP transform + reconstruction for odd images
        self.f = self.f[0:-1,0:-1]
        [ewtc,mfb,bounds] = ewt2d.ewt2dLP(self.f, self.params)
        recon = ewt2d.iewt2dLP(ewtc,mfb)
        self.assertTrue(np.sum((recon-self.f)**2) < 10**(-10))
    def testewt2dRidgeletEven(self):
        #Tests empirical ridgelet transform + reconstruction for odd images
        [ewtc,mfb,bounds] = ewt2d.ewt2dRidgelet(self.f, self.params)
        recon = ewt2d.iewt2dRidgelet(ewtc,mfb)
        self.assertTrue(np.sum((recon-self.f)**2) < 10**(-3))
    def testewt2dRidgeletOdd(self):
        #Tests empirical ridgelet transform + reconstruction for odd images
        self.f = self.f[0:-1,0:-1]
        [ewtc,mfb,bounds] = ewt2d.ewt2dRidgelet(self.f, self.params)
        recon = ewt2d.iewt2dRidgelet(ewtc,mfb,1)
        self.assertTrue(np.sum((recon-self.f)**2) < 10**(-3))
    def testppfft(self):
        #Tests pseudo-polar fourier transform 
        ppff = ewt2d.ppfft(self.f)
        recon = ewt2d.ippfft(ppff)
        self.assertTrue(np.sum((recon-self.f)**2) < 10**(-10))
        
class EWT_UnitTest_Filters(unittest.TestCase):
    def test_ewt2d_curveletWavelet1(self):
        #Tests curvelet filterbank construction in typical example
        h = 127;
        w = 127;
        option = 1;
        bounds_scales = [.5,.9,1.3,3] #between 0 and pi
        bounds_angles = [-2,-.5,.3] #between -3pi/4(-2.356+ and pi/4 (.7854)
        mfb = ewt2d.ewt2d_curveletFilterbank(bounds_scales,bounds_angles,h,w,option)
        tmp = np.zeros([h,w])
        for i in range(0,len(mfb)):
            for j in range(0,len(mfb[i])):
                tmp += mfb[i][j]**2
        
        self.assertAlmostEqual(np.max(tmp),1.0)
        self.assertAlmostEqual(np.min(tmp),1.0)
    def test_ewt2d_curveletWavelet2(self):
        #Tests curvelet filterbank construction if given single angular bound
        h = 127;
        w = 127;
        option = 1;
        bounds_scales = [.5,.9,1.3,3] #between 0 and pi
        bounds_angles = [.3] #between -3pi/4(-2.356+ and pi/4 (.7854)
        mfb = ewt2d.ewt2d_curveletFilterbank(bounds_scales,bounds_angles,h,w,option)
        tmp = np.zeros([h,w])
        for i in range(0,len(mfb)):
            for j in range(0,len(mfb[i])):
                tmp += mfb[i][j]**2
        self.assertAlmostEqual(np.max(tmp),1.0)
        self.assertAlmostEqual(np.min(tmp),1.0)
    def test_ewt2d_curveletWavelet3(self):
        #Tests curvelet filterbank construction if given two scales that are very close
        h = 127;
        w = 127;
        option = 1;
        epsilon = 0.0000000001
        bounds_scales = [.5,.9,1.3,1.3+epsilon,3] #between 0 and pi
        bounds_angles = [-2,-.5,.3] #between -3pi/4(-2.356+ and pi/4 (.7854)
        mfb = ewt2d.ewt2d_curveletFilterbank(bounds_scales,bounds_angles,h,w,option)
        tmp = np.zeros([h,w])
        for i in range(0,len(mfb)):
            for j in range(0,len(mfb[i])):
                tmp += mfb[i][j]**2
        self.assertAlmostEqual(np.max(tmp),1.0)
        self.assertAlmostEqual(np.min(tmp),1.0)
    def test_ewt2d_curveletWavelet4(self):
        #Tests curvelet filterbank construction if given two angles that are very close
        h = 127;
        w = 127;
        option = 1;
        bounds_scales = [.5,.9,1.3,3] #between 0 and pi
        epsilon = 0.0000000001
        bounds_angles = [-2,-.5+epsilon,.3] #between -3pi/4(-2.356+ and pi/4 (.7854)
        mfb = ewt2d.ewt2d_curveletFilterbank(bounds_scales,bounds_angles,h,w,option)
        tmp = np.zeros([h,w])
        for i in range(0,len(mfb)):
            for j in range(0,len(mfb[i])):
                tmp += mfb[i][j]**2
        self.assertAlmostEqual(np.max(tmp),1.0)
        self.assertAlmostEqual(np.min(tmp),1.0)
    def test_ewt2d_curveletWavelet5(self):
        #Tests curvelet filterbank construction if given no scales
        h = 127;
        w = 127;
        option = 1;
        bounds_scales = [] #between 0 and pi
        bounds_angles = [-2,-.5,.3] #between -3pi/4(-2.356+ and pi/4 (.7854)
        mfb = ewt2d.ewt2d_curveletFilterbank(bounds_scales,bounds_angles,h,w,option)
        tmp = np.zeros([h,w])
        for i in range(0,len(mfb)):
            for j in range(0,len(mfb[i])):
                tmp += mfb[i][j]**2     
        self.assertAlmostEqual(np.max(tmp),1.0)
        self.assertAlmostEqual(np.min(tmp),1.0)
    def test_ewt2d_curveletWavelet6(self):
        #Tests curvelet filterbank construction if given no angles 
        h = 127;
        w = 127;
        option = 1;
        bounds_scales = [.5,.9,1.3,3] #between 0 and pi
        bounds_angles = [] #between -3pi/4(-2.356+ and pi/4 (.7854)
        mfb = ewt2d.ewt2d_curveletFilterbank(bounds_scales,bounds_angles,h,w,option)

        tmp = np.zeros([h,w])
        for i in range(0,len(mfb)):
            for j in range(0,len(mfb[i])):
                tmp += mfb[i][j]**2
        self.assertAlmostEqual(np.max(tmp),1.0)
        self.assertAlmostEqual(np.min(tmp),1.0)
    def test_ewt2d_curveletWavelet7(self):
        #Tests curvelet filterbank construction if given scale out of bounds (below)
        h = 127;
        w = 127;
        option = 1;
        bounds_scales = [-.5,.9,1.3,3] #between 0 and pi
        bounds_angles = [-2,-.5,.3] #between -3pi/4(-2.356+ and pi/4 (.7854)
        mfb = ewt2d.ewt2d_curveletFilterbank(bounds_scales,bounds_angles,h,w,option)
        tmp = np.zeros([h,w])
        for i in range(0,len(mfb)):
            for j in range(0,len(mfb[i])):
                tmp += mfb[i][j]**2
        self.assertAlmostEqual(np.max(tmp),1.0)
        self.assertAlmostEqual(np.min(tmp),1.0)    
    def test_ewt2d_curveletWavelet8(self):
        #Tests curvelet filterbank construction if given scale out of bounds (above)
        h = 127;
        w = 127;
        option = 1;
        bounds_scales = [.5,.9,1.3,3.4] #between 0 and pi
        bounds_angles = [-2,-.5,.3] #between -3pi/4(-2.356+ and pi/4 (.7854)
        mfb = ewt2d.ewt2d_curveletFilterbank(bounds_scales,bounds_angles,h,w,option)
        tmp = np.zeros([h,w])
        for i in range(0,len(mfb)):
            for j in range(0,len(mfb[i])):
                tmp += mfb[i][j]**2
        self.assertAlmostEqual(np.max(tmp),1.0)
        self.assertAlmostEqual(np.min(tmp),1.0)
    def test_ewt2d_curveletWavelet9(self):
        #Tests curvelet filterbank construction if given angle out of bounds (below)
        h = 127;
        w = 127;
        option = 1;
        bounds_scales = [.5,.9,1.3,3] #between 0 and pi
        bounds_angles = [-3,-2,-.5,.3] #between -3pi/4(-2.356+ and pi/4 (.7854)
        mfb = ewt2d.ewt2d_curveletFilterbank(bounds_scales,bounds_angles,h,w,option)
        tmp = np.zeros([h,w])
        for i in range(0,len(mfb)):
            for j in range(0,len(mfb[i])):
                tmp += mfb[i][j]**2
        self.assertAlmostEqual(np.max(tmp),1.0)
        self.assertAlmostEqual(np.min(tmp),1.0)
    def test_ewt2d_curveletWavelet10(self):
        #Tests curvelet filterbank construction if given angle out of bounds (above)
        h = 127;
        w = 127;
        option = 1;
        bounds_scales = [.5,.9,1.3,3] #between 0 and pi
        bounds_angles = [-2,-.5,.3,1.0] #between -3pi/4(-2.356+ and pi/4 (.7854)
        mfb = ewt2d.ewt2d_curveletFilterbank(bounds_scales,bounds_angles,h,w,option)
        tmp = np.zeros([h,w])
        for i in range(0,len(mfb)):
            for j in range(0,len(mfb[i])):
                tmp += mfb[i][j]**2
        self.assertAlmostEqual(np.max(tmp),1.0)
        self.assertAlmostEqual(np.min(tmp),1.0)
    def test_ewt2d_LPWavelet1(self):
        #Tests LP filterbank construction in normal circumstances
        h = 127;
        w = 127;
        bounds_scales = [.5,.9,1.3,3] #between 0 and pi
        mfb = ewt2d.ewt2d_LPFilterbank(bounds_scales,h,w)
        tmp = np.zeros([h,w])
        for i in range(0,len(mfb)):
            for j in range(0,len(mfb[i])):
                tmp += mfb[i][j]**2
        self.assertAlmostEqual(np.max(tmp),1.0)
        self.assertAlmostEqual(np.min(tmp),1.0)
    def test_ewt2d_LPWavelet2(self):
        #Tests LP filterbank construction if no scales detected
        h = 127;
        w = 127;
        bounds_scales = [] #between 0 and pi
        mfb = ewt2d.ewt2d_LPFilterbank(bounds_scales,h,w)
        tmp = np.zeros([h,w])
        for i in range(0,len(mfb)):
            for j in range(0,len(mfb[i])):
                tmp += mfb[i][j]**2
        self.assertAlmostEqual(np.max(tmp),1.0)
        self.assertAlmostEqual(np.min(tmp),1.0)
    #Tests LP filterbank construction in normal circumstances
        h = 127;
        w = 127;
        bounds_scales = [.5,.9,1.3,3] #between 0 and pi
        mfb = ewt2d.ewt2d_LPFilterbank(bounds_scales,h,w)
        tmp = np.zeros([h,w])
        for i in range(0,len(mfb)):
            for j in range(0,len(mfb[i])):
                tmp += mfb[i][j]**2
        self.assertAlmostEqual(np.max(tmp),1.0)
        self.assertAlmostEqual(np.min(tmp),1.0)
    
class EWT_UnitTest_Boundaries(unittest.TestCase):        
    def test_boundaries_localmin1(self):
        #basic test for localmin
        inp = np.array([1,0,1])
        out = np.array([0,1,0],'bool')
        self.assertTrue(np.all(boundaries.localmin(inp) == out))
    def test_boundaries_localmin2(self):
        #boundary test for localmin
        inp = np.array([0,1,2,0,1,0])
        out = np.array([0,0,0,1,0,0],'bool')
        self.assertTrue(np.all(boundaries.localmin(inp) == out))
    def test_boundaries_localmin3(self):
        #even plateau test for localmin
        inp = np.array([2, 1, 0, 0, 1, 2])
        out = np.array([0 ,0, 1, 0, 0, 0],'bool')
        self.assertTrue(np.all(boundaries.localmin(inp) == out))    
    def test_boundaries_localmin4(self):
        #odd plateau test for localmin
        inp = np.array([2, 1, 0, 0, 0, 1, 2])
        out = np.array([0 ,0, 0, 1, 0, 0, 0],'bool')
        self.assertTrue(np.all(boundaries.localmin(inp) == out))
    def test_boundaries_localmin5(self):
        #boundary even plateau test for localmin
        inp = np.array([0, 0, 1, 2, 0, 0])
        out = np.array([0, 1, 0, 0, 1, 0],'bool')
        self.assertTrue(np.all(boundaries.localmin(inp) == out))
    def test_boundaries_localmin6(self):
        #boundary odd plateau test for localmin
        inp = np.array([0, 0, 0, 1, 2, 0, 0, 0])
        out = np.array([0 ,1, 0, 0, 0, 0, 1, 0],'bool')
        self.assertTrue(np.all(boundaries.localmin(inp) == out))
    def test_boundaries_localmin7(self):
        #saddle test for localmin
        inp = np.array([3, 2, 2, 2, 1, 0, 2])
        out = np.array([0 ,0, 0, 0, 0, 1, 0],'bool')
        self.assertTrue(np.all(boundaries.localmin(inp) == out))
    def test_boundaries_localmin8(self):
        #odd flat test for localmin
        inp = np.array([0, 0, 0])
        out = np.array([0, 1, 0],'bool')
        self.assertTrue(np.all(boundaries.localmin(inp) == out))
    def test_boundaries_localmin9(self):
        #even flat test for localmin
        inp = np.array([0, 0, 0, 0])
        out = np.array([0 ,1, 0, 0],'bool')
        self.assertTrue(np.all(boundaries.localmin(inp) == out))
    def test_boundaries_localmin10(self):
        #epsilon test for localmin
        epsilon = .00000000000000001
        inp = np.array([2, 1, 0-epsilon, 0+epsilon, 0-epsilon, 1, 2])
        out = np.array([0 ,0, 1, 0, 1, 0, 0],'bool')
        self.assertTrue(np.all(boundaries.localmin(inp) == out))
if __name__ == '__main__':
    unittest.main()
