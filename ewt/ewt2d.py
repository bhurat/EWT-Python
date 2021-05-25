# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 13:59:13 2020

@author: bazzz
"""
import numpy as np
from ewt.boundaries import *
from ewt.ewt1d import *


"""
ewt2dTensor(f,params)
Given an image, function performs the 2D Tensor empirical wavelet transform - 
This is a separable approach that contructs a filter bank based on detected 
rectangular supports
Input:
    f           - 2D array containing signal
    params      - parameters for EWT (see utilities)
Output:
    ewtc        - empirical wavelet coefficients
    mfb_row     - constructed Tensor empirical wavelet filter bank for rows
    mfb_col     - constructed Tensor empirical wavelet filter bank for columns
    bounds_row  - detected row boundaries in [0,pi]
    bounds_col  - detected column boundaries in [0,pi]
Author: Basile Hurat, Jerome Gilles"""
def ewt2dTensor(f,params):
    ff = np.fft.fft(f,axis = 0) #take 1D fft along columns
    meanfft = np.sum(np.abs(ff),axis = 1)/ff.shape[0] #take average of magnitude
    bounds_col = ewt_boundariesDetect(meanfft,params)
    bounds_col = bounds_col*2*np.pi/len(meanfft)
    
    ewtc_col = []
    mfb_col = ewt_LP_Filterbank(bounds_col,len(meanfft),params.real) #construct filter bank
    for i in range(0,len(mfb_col)):
        filter_col = np.tile(mfb_col[i],[f.shape[1],1]).T #repeat down
        ewtc_col.append(np.real(np.fft.ifft(filter_col*ff,axis = 0))) #get coefficients,
    
    # REPEAT WITH ROWS #
    ff = np.fft.fft(f,axis = 1) #take 1D fft along rows
    meanfft = np.mean(np.abs(ff.T),axis = 1) 
    bounds_row = ewt_boundariesDetect(meanfft,params)
    bounds_row = bounds_row*2*np.pi/len(meanfft)
    
    ewtc = []
    mfb_row = ewt_LP_Filterbank(bounds_row,len(meanfft),params.real)
    for i in range(0,len(mfb_row)):
        ewtc_row = []
        filter_row = np.tile(mfb_row[i],[f.shape[0],1])
        for j in range(0,len(mfb_col)):
            ff = np.fft.fft(ewtc_col[j])
            ewtc_row.append(np.real(np.fft.ifft(filter_row*ff,axis = 1)))
        ewtc.append(ewtc_row)
            
    return [ewtc, mfb_row, mfb_col,bounds_row,bounds_col]

"""
iewt2dTensor(ewtc,mfb_row,mfb_col)
Performs the inverse Tensor 2D empirical wavelet transform given the Tensor EWT
ceofficients and filters
Input:
    ewtc    - Tensor empirical wavelet coefficients
    mfb_row - corresponding Tensor empirical wavelet filter bank for rows
    mfb_col - corresponding Tensor empirical wavelet filter bank for columns
Output:
    img     - reconstructed image
Author: Basile Hurat, Jerome Gilles"""  
def iewt2dTensor(ewtc,mfb_row,mfb_col):
    [h,w]= ewtc[0][0].shape
    ewt_col = []
    for i in range(0,len(mfb_col)):
        for j in range(0,len(mfb_row)):
            ff = np.fft.fft(ewtc[j][i])
            filter_row = np.tile(mfb_row[j],[h, 1])
            if j == 0:
                ewt_col.append(np.zeros([h,w]))
            ewt_col[i] += np.real(np.fft.ifft(filter_row*ff))
    
    for i in range(0,len(mfb_col)):
        ff = np.fft.fft(ewt_col[i].T);
        if i == 0:
            img = np.zeros([h,w])
        
        filter_col = np.tile(mfb_col[i],[w, 1])
        img = img + np.real(np.fft.ifft(filter_col*ff)).T
    return img

"""
ewt2dLP(f,params)
Given an image, function performs the 2D Littlewood-Paley empirical wavelet 
transform - This is an approach that contructs a filter bank of anulli based on 
detected scales. 
Input:
    f               - vector x as an input
    params          - parameters for EWT (see utilities)
Output:
    ewtc            - empirical wavelet coefficients
    mfb             - constructed Curvelet empirical wavelet filter bank
    bounds_scales   - detected scale boundaries in [0,pi]
Author: Basile Hurat, Jerome Gilles"""
def ewt2dLP(f,params):
    [h,w] = f.shape
    #get boundaries
    ppff = ppfft(f)
    meanppff = np.fft.fftshift(np.mean(np.abs(ppff),axis = 1))
    bounds_scales = ewt_boundariesDetect(meanppff,params)
    bounds_scales *= np.pi/np.ceil((len(meanppff)/2))
   
    #construct filter bank
    mfb = ewt2d_LPFilterbank(bounds_scales,h,w)
    
    #filter out coefficients
    ff = np.fft.fft2(f);
    ewtc = []
    for i in range(0,len(mfb)):
        ewtc.append(np.real(np.fft.ifft2(mfb[i]*ff)))
    return [ewtc, mfb, bounds_scales]

"""
iewt2dLP(ewtc,mfb)
Performs the inverse Littlewood-Paley 2D empirical wavelet transform given the 
Littlewood-Paley EWT ceofficients and filters
Input:
    ewtc    - Littlewood-Paley empirical wavelet coefficients
    mfb     - corresponding Littlewood-Paley filter bank
Output:
    recon   - reconstructed image
Author: Basile Hurat, Jerome Gilles"""  
def iewt2dLP(ewtc,mfb):
    recon = np.fft.fft2(ewtc[0])*mfb[0]
    for i in range(1,len(mfb)):
        recon += np.fft.fft2(ewtc[i])*mfb[i]
    recon = np.real(np.fft.ifft2(recon))
    return recon
        
"""
ewt2d_LPFilterbank(bounds_scales,h,w)
Constructs the Littlewood-Paley 2D EWT filter bank with filters of size [h,w]
based on a set of detected scales
Input:
    bounds_scales   - detected scale bounds in range [0,pi]
    h               - desired height of filters
    w               - desired width of filters
Output:
    mfb             - Littlewood-Paley 2D EWT filter bank
Author: Basile Hurat, Jerome Gilles"""  
def ewt2d_LPFilterbank(bounds_scales,h,w):
    if h%2 == 0:
        h += 1
        h_extended = 1
    else:
        h_extended = 0
    if w%2 == 0:
        w += 1
        w_extended = 1
    else:
        w_extended = 0
    #First, we calculate gamma for scales
    gamma_scales = np.pi
    for k in range(0,len(bounds_scales)-1):
        r = (bounds_scales[k+1] - bounds_scales[k])/(bounds_scales[k+1] + bounds_scales[k])
        if r < gamma_scales and r > 1e-16:
            gamma_scales = r
    
    r = (np.pi - bounds_scales[-1])/(np.pi + bounds_scales[-1]) #check last bound
    if r < gamma_scales and r > 1e-16:
        gamma_scales = r
    if gamma_scales > bounds_scales[0]:     #check first bound
        gamma_scales = bounds_scales[0]
    gamma_scales *= (1 - 1/max(h,w)) #guarantees that we have strict inequality
    radii = np.zeros([h,w])
    
    h_center = h//2 + 1; w_center = w//2+1
    for i in range(0,h):
        for j in range(0,w):
            ri = (i+1.0 - h_center)*np.pi/h_center
            rj = (j+1.0 - w_center)*np.pi/w_center
            radii[i,j] = np.sqrt(ri**2 + rj**2)
    
    mfb = []
    mfb.append(ewt2d_LPscaling(radii,bounds_scales[0],gamma_scales))
    for i in range(0,len(bounds_scales)-1):
        mfb.append(ewt2d_LPwavelet(radii,bounds_scales[i],bounds_scales[i+1],gamma_scales))
    mfb.append(ewt2d_LPwavelet(radii,bounds_scales[-1],2*np.pi,gamma_scales))
    
    if h_extended == 1: #if we extended the height of the image, trim
        h -= 1
        for i in range(0,len(mfb)):
            mfb[i] = mfb[i][0:-1,:]
    if w_extended == 1: #if we extended the width of the image, trim
        w -= 1
        for i in range(0,len(mfb)):
            mfb[i] = mfb[i][:,0:-1]
    #invert the fftshift since filters are centered
    for i in range(0,len(mfb)):
        mfb[i] = np.fft.ifftshift(mfb[i])
            
    #Resymmetrize for even images
    if h_extended == 1:
        s = np.zeros(w)
        if w%2 == 0:
            mfb[-1][h//2, 1:w//2] += mfb[-1][h//2, -1:w//2:-1]
            mfb[-1][h//2, w//2+1:] = mfb[-1][h//2, w//2-1:0:-1]
            s += mfb[-1][h//2,:]**2
            #normalize for tight frame
            mfb[-1][h//2, 1:w//2] /= np.sqrt(s[1:w//2])
            mfb[-1][h//2, w//2+1:] /= np.sqrt(s[w//2+1:])
        else:
            mfb[-1][h//2, 0:w//2] += mfb[-1][h//2, -1:w//2:-1]
            mfb[-1][h//2, w//2+1:] = mfb[-1][h//2, w//2-1::-1]
            s += mfb[-1][h//2,:]**2
            #normalize for tight frame
            mfb[-1][h//2,0:w//2]  /= np.sqrt(s[0:w//2])
            mfb[-1][h//2,w//2+1:] /= np.sqrt(s[w//2+1:])
    if w_extended == 1:
        s = np.zeros(h)
        if h%2 == 0:
            mfb[-1][1:h//2, w//2] += mfb[-1][-1:h//2:-1, w//2]
            mfb[-1][h//2+1:, w//2] = mfb[-1][h//2-1:0:-1, w//2]
            s += mfb[-1][:, w//2]**2
            #normalize for tight frame
            mfb[-1][1:h//2, w//2] /= np.sqrt(s[1:h//2])
            mfb[-1][h//2+1:, w//2] /= np.sqrt(s[h//2+1:]) 
        else:
            mfb[-1][0:h//2, w//2] += mfb[-1][-1:h//2:-1, w//2]
            mfb[-1][h//2+1:, w//2] = mfb[-1][h//2-1::-1, w//2]
            s += mfb[-1][:, w//2]**2
            #normalize for tight frame
            mfb[-1][0:h//2, w//2] /= s[0:h//2]
            mfb[-1][h//2+1:, w//2] /= s[h//2+1:]
    return mfb

"""
ewt2d_LPscaling(radii,bound0,gamma)
Constructs the empirical Littlewood-Paley scaling function (circle)
Input:
    radii   - reference image where pixels are equal to their distance from 
            center
    bound0   - first radial bound
    gamma   - detected scale gamma to guarantee tight frame
Output:
    scaling - resulting empirical Littlewood-Paley scaling function
Author: Basile Hurat, Jerome Gilles""" 
def ewt2d_LPscaling(radii,bound0,gamma):
    an = 1/(2*gamma*bound0) 
    mbn = (1 - gamma)*bound0 # inner circle up to beginning of transtion
    pbn = (1 + gamma)*bound0 #end of transition
    
    scaling = 0*radii #initiate w/ zeros
    scaling[radii < mbn] = 1
    scaling[radii >= mbn] = np.cos(np.pi*ewt_beta(an*(radii[radii>=mbn] - mbn))/2)
    scaling[radii > pbn] = 0
    return scaling

"""
ewt2d_LPwavelet(radii,bound1,bound2,gamma)
Constructs the empirical Littlewood-Paley wavelet function (annulus)
Input:
    radii   - reference image where pixels are equal to their distance from 
            center
    bound1  - lower radial bound
    bound2  - upper radial bound
    gamma   - detected gamma to guarantee tight frame
Output:
    wavelet - resulting empirical Littlewood-Paley wavelet
Author: Basile Hurat, Jerome Gilles""" 
def ewt2d_LPwavelet(radii,bound1,bound2,gamma):
    wan = 1/(2*gamma*bound1) #scaling factor
    wam = 1/(2*gamma*bound2) 
    wmbn = (1 - gamma)*bound1 #beginning of lower transition
    wpbn = (1 + gamma)*bound1 #end of lower transition
    wmbm = (1 - gamma)*bound2  #beginning of upper transition
    wpbm = (1 + gamma)*bound2 #end of upper transition
    
    wavelet = 0*radii #initialize w/ zeros
    inside = (radii > wmbn)*(radii < wpbm)
    wavelet[inside] = 1.0 #set entire angular wedge equal to 1
    temp = inside*(radii >= wmbm)*(radii <= wpbm) #upper transition
    wavelet[temp] *= np.cos(np.pi*ewt_beta(wam*(radii[temp]-wmbm))/2)
    temp = inside*(radii >= wmbn)*(radii <= wpbn) #lower transition
    wavelet[temp] *= np.sin(np.pi*ewt_beta(wan*(radii[temp]-wmbn))/2)
    return wavelet

"""
ewt2dRidgelet(f,params)
Given an image, function performs the 2D Littlewood-Paley empirical wavelet 
transform - This is an approach that contructs a filter bank of anulli based on 
detected scales. 
Input:
    f               - vector x as an input
    params          - parameters for EWT (see utilities)
Output:
    ewtc            - empirical wavelet coefficients
    mfb             - constructed Curvelet empirical wavelet filter bank
    bounds_scales   - detected scale boundaries in [0,pi]
Author: Basile Hurat, Jerome Gilles"""
def ewt2dRidgelet(f,params):
    [h,w] = f.shape
    #get boundaries
    ppff = ppfft(f)
    meanppff = np.fft.fftshift(np.mean(np.abs(ppff),axis = 1))
    bounds_scales = ewt_boundariesDetect(meanppff,params)
    bounds_scales *= np.pi/np.ceil((len(meanppff)/2))
    
    #Construct 1D filterbank
    mfb_1d = ewt_LP_Filterbank(bounds_scales,ppff.shape[0],params.real)
    
    #filter out coefficients
    mfb = []
    ewtc = []
    for i in range(0,len(mfb_1d)):
        mfb.append(np.tile(mfb_1d[i],[ppff.shape[1],1]).T)
        ewtc.append(np.real(np.fft.ifft(mfb[i]*np.fft.fftshift(ppff,0),axis = 0)))
    return [ewtc, mfb, bounds_scales]
"""
iewt2dRidgelet(ewtc,mfb)
Performs the inverse Curvelet 2D empirical wavelet transform given the Curvelet
EWT ceofficients and filters
Input:
    ewtc    - Curvelet empirical wavelet coefficients
    mfb     - corresponding Curvelet filter bank
Output:
    recon   - reconstructed image
Author: Basile Hurat, Jerome Gilles""" 

def iewt2dRidgelet(ewtc,mfb,isOdd = 0):
   ppff=1j*np.zeros(ewtc[0].shape);
   for i in range(0,len(mfb)):
       ppff += np.fft.fftshift(np.fft.fft(ewtc[i],axis = 0)*mfb[i],0)
   recon = np.real(ippfft(ppff,1e-10))
   if isOdd:
       recon = recon[0:-1,0:-1]
   return recon

"""
ewt2dCurvelet(f,params)
Given an image, function performs the 2D Curvelet empirical wavelet transform - 
This is an approach that contructs a filter bank of polar wedges based on 
detected scales and orientations. There are three options
    Option 1 - Detects scales and orientations separately
    Option 2 - Detects scales first, and then orientations for each scale
    Option 3 - Detects orientations first, and then scales for each orientation
Input:
    f               - vector x as an input
    params          - parameters for EWT (see utilities)
Output:
    ewtc            - empirical wavelet coefficients
    mfb             - constructed Curvelet empirical wavelet filter bank
    bounds_scales  - detected scale boundaries in [0,pi]
    bounds_angles  - detected angle boundaries in [-3pi/4,pi/4]
Author: Basile Hurat, Jerome Gilles"""
def ewt2dCurvelet(f,params):
    [h,w] = f.shape
    ppff = ppfft(f)
    
    if params.option == 1:
    #Option 1: Computes scales and angles independently    
        #Begin with scales
        meanppff = np.fft.fftshift(np.mean(np.abs(ppff),axis = 1))
        bounds_scales = ewt_boundariesDetect(meanppff,params)
        bounds_scales *= np.pi/np.ceil((len(meanppff)/2))
        
        #Then do with angles
        meanppff = np.mean(np.abs(ppff),axis = 0)
        bounds_angles = ewt_boundariesDetect(meanppff,params)
        bounds_angles = bounds_angles*np.pi/np.ceil((len(meanppff)/2)) - np.pi*.75
    
    elif params.option == 2:
        #Option 2: Computes Scales first, then angles 
        meanppff = np.fft.fftshift(np.mean(np.abs(ppff),axis = 1))
        bounds_scales = ewt_boundariesDetect(meanppff,params)
        bounds_angles = []
        for i in range(0,len(bounds_scales)-1):
            meanppff = np.mean(np.abs(ppff[int(bounds_scales[i]):int(bounds_scales[i+1]+1),:]),axis = 0)
            bounds =  ewt_boundariesDetect(meanppff,params)
            #append
            bounds_angles.append(bounds*np.pi/np.ceil((len(meanppff)/2)) - np.pi*.75)
            
        #Do last linterval
        meanppff = np.mean(np.abs(ppff[int(bounds_scales[-1]):,:]),axis = 0)
        bounds =  ewt_boundariesDetect(meanppff,params)
        #append
        bounds_angles.append(bounds*np.pi/np.ceil((len(meanppff)/2)) - np.pi*.75)
        #normalize scale bounds
        bounds_scales *= np.pi/np.ceil((len(meanppff)/2))
        
    elif params.option == 3:
        #Option 3: Computes angles, then scales
        bounds_scales = []
        #Get first scale
        meanppff = np.fft.fftshift(np.mean(np.abs(ppff),axis = 1))
        LL = len(meanppff)//2
        #bound0 = ewt_boundariesDetect(meanppff[LL:],params)[0]
        bound0 = ewt_boundariesDetect(meanppff,params)[0]
        bounds_scales.append([bound0*np.pi/LL])
        bound0 = int(bound0)
        #Compute mean-pseudo-polar fft for angles, excluding first scale to 
        #find angle bounds
        #meanppff = np.mean(np.abs(ppff[ppff.shape[0]//2+bound0:,:]),0)
        meanppff = np.mean(np.abs(ppff[bound0:-bound0,:]),axis = 0)
        bounds_theta = ewt_boundariesDetect(meanppff,params, sym = 0)
        bounds_angles = (bounds_theta-1)*np.pi/len(meanppff)-0.75*np.pi
        bounds_theta = bounds_theta.astype(int)
        #Now we find scale bounds at each angle
        for i in range(0,len(bounds_theta)-1):
            #meanppff = np.mean(np.abs(ppff[LL+bound0:,bounds_theta[i]:bounds_theta[i+1]+1]),1)
            meanppff = np.fft.fftshift(np.mean(np.abs(ppff[bound0:-bound0,bounds_theta[i]:bounds_theta[i+1]+1]),1))
            bounds = ewt_boundariesDetect(meanppff,params)
            bounds_scales.append((bounds+bound0)*np.pi/LL)
        
        #and also for the last angle
        #meanppff = np.mean(np.abs(ppff[LL+bound0:,bounds_theta[-1]:]),1)
        #meanppff += np.mean(np.abs(ppff[LL+bound0:,1:bounds_theta[0]+1]),1)
        meanppff = np.mean(np.abs(ppff[bound0:-bound0,bounds_theta[-1]:]),1)
        meanppff += np.mean(np.abs(ppff[bound0:-bound0,1:bounds_theta[0]+1]),1)
        
        params.spectrumRegularize = 'closing'
        bounds = ewt_boundariesDetect(np.fft.fftshift(meanppff),params)
        bounds_scales.append((bounds+bound0)*np.pi/LL)
    else:
        print('invalid option')
        return -1
    #Once bounds are found, construct filter bank, take fourier transform of 
    #image, and filter
    mfb = ewt2d_curveletFilterbank(bounds_scales,bounds_angles,h,w,params.option)
    ff = np.fft.fft2(f)
    ###ewtc = result!
    ewtc = []
    for i in range(0,len(mfb)):
        ewtc_scales = []
        for j in range(0,len(mfb[i])):
            ewtc_scales.append(np.real(np.fft.ifft2(mfb[i][j]*ff)))
        ewtc.append(ewtc_scales)
    return [ewtc, mfb, bounds_scales, bounds_angles]

"""
iewt2dCurvelet(ewtc,mfb)
Performs the inverse Curvelet 2D empirical wavelet transform given the Curvelet
EWT ceofficients and filters
Input:
    ewtc    - Curvelet empirical wavelet coefficients
    mfb     - corresponding Curvelet filter bank
Output:
    recon   - reconstructed image
Author: Basile Hurat, Jerome Gilles"""  
def iewt2dCurvelet(ewtc,mfb):
    recon = np.fft.fft2(ewtc[0][0])*mfb[0][0]
    for i in range(1,len(mfb)):
        for j in range(0,len(mfb[i])):
            recon += np.fft.fft2(ewtc[i][j])*mfb[i][j]
    recon = np.real(np.fft.ifft2(recon))
    return recon
        
"""
ewt_curveletFilterbank(bounds_scales,bounds_angles,h,w,option)
Constructs the Curvelet 2D empirical wavelet filter bank 
Input:
    bounds_scales   - detected scale boundaries
    bounds_angles   - detected orientation boundaries
    h               - desired height of filters
    w               - desired width of filters
    option          - Option for curvelet
                        1 for separate detection of scales and orientations
                        2 for detection of scales and then orientations 
                        3 for detection of orientations and then scales
Output:
    mfb             - resulting filter bank 
Author: Basile Hurat, Jerome Gilles"""      
def ewt2d_curveletFilterbank(bounds_scales,bounds_angles,h,w,option):

    if h%2 == 0:
        h += 1
        h_extended = 1
    else:
        h_extended = 0
    if w%2 == 0:
        w += 1
        w_extended = 1
    else:
        w_extended = 0
    
    
    if option == 1:
        #Scales and angles detected separately
        
        #First, we calculate gamma for scales
        gamma_scales = np.pi
        for k in range(0,len(bounds_scales)-1):
            r = (bounds_scales[k+1] - bounds_scales[k])/(bounds_scales[k+1] + bounds_scales[k])
            if r < gamma_scales and r > 1e-16:
                gamma_scales = r
        
        r = (np.pi - bounds_scales[-1])/(np.pi + bounds_scales[-1]) #check last bound
        if r < gamma_scales and r > 1e-16:
            gamma_scales = r
        if gamma_scales > bounds_scales[0]:     #check first bound
            gamma_scales = bounds_scales[0]
        gamma_scales *= (1 - 1/max(h,w)) #guarantees that we have strict inequality
        
        #Get gamma for angles
        gamma_angles = 2*np.pi
        for k in range(0,len(bounds_angles)-1):
            r = (bounds_angles[k+1] - bounds_angles[k])/2
            if r < gamma_angles and r > 1e-16:
                gamma_angles = r
        r = (bounds_angles[0] + np.pi - bounds_angles[-1])/2 #check extreme bounds (periodic)
        if r < gamma_angles and r > 1e-16:
            gamma_angles = r
        gamma_angles *= (1 - 1/max(h,w)) #guarantees that we have strict inequality    
        
        #construct matrices representing radius and angle value of each pixel
        radii = np.zeros([h,w])
        theta = np.zeros([h,w])
        h_center = h//2 + 1; w_center = w//2+1
        for i in range(0,h):
            for j in range(0,w):
                ri = (i+1.0 - h_center)*np.pi/h_center
                rj = (j+1.0 - w_center)*np.pi/w_center
                radii[i,j] = np.sqrt(ri**2 + rj**2)
                theta[i,j] = np.arctan2(ri,rj)
                if theta[i,j] <-.75*np.pi:
                    theta[i,j] += 2*np.pi
        
        mfb = []
        #construct scaling
        mfb.append([ewt2d_curveletScaling(radii,bounds_scales[0],gamma_scales)])
        
        #construct angular wedges for all but last scales
        for i in range(0,len(bounds_scales)-1):
            mfb_scale = []           
            for j in range(0,len(bounds_angles)-1):
                mfb_scale.append(ewt2d_curveletWavelet(theta,radii,
                                                       bounds_angles[j],bounds_angles[j+1],
                                                       bounds_scales[i],bounds_scales[i+1],
                                                       gamma_angles,gamma_scales))
            mfb_scale.append(ewt2d_curveletWavelet(theta,radii,
                                                       bounds_angles[-1],bounds_angles[0]+np.pi,
                                                       bounds_scales[i],bounds_scales[i+1],
                                                       gamma_angles,gamma_scales))
            mfb.append(mfb_scale)
        #construct angular wedges for last scales
    
        mfb_scale = []            
        for j in range(0,len(bounds_angles)-1):
            mfb_scale.append(ewt2d_curveletWavelet(theta,radii,
                                                   bounds_angles[j],bounds_angles[j+1],
                                                   bounds_scales[-1],np.pi*2,
                                                   gamma_angles,gamma_scales))
        mfb_scale.append(ewt2d_curveletWavelet(theta,radii,
                                                   bounds_angles[-1],bounds_angles[0]+np.pi,
                                                   bounds_scales[-1],np.pi*2,
                                                   gamma_angles,gamma_scales))
        mfb.append(mfb_scale)
    elif option == 2:
        #Scales detected first, and then angles
        
        #Get gamma for scales
        gamma_scales = np.pi
        for k in range(0,len(bounds_scales)-1):
            r = (bounds_scales[k+1] - bounds_scales[k])/(bounds_scales[k+1] + bounds_scales[k])
            if r < gamma_scales and r > 1e-16:
                gamma_scales = r
        
        r = (np.pi - bounds_scales[-1])/(np.pi + bounds_scales[-1]) #check last bound
        if r < gamma_scales and r > 1e-16:
            gamma_scales = r
        if gamma_scales > bounds_scales[0]:     #check first bound
            gamma_scales = bounds_scales[0]
        gamma_scales *= (1 - 1/max(h,w)) #guarantees that we have strict inequality
        
        #Get gammas for angles
        gamma_angles = 2*np.pi*np.ones(len(bounds_scales))
        for i in range(0,len(gamma_angles)):
            for k in range(0,len(bounds_angles[i])-1):
                r = (bounds_angles[i][k+1] - bounds_angles[i][k])/2
                if r < gamma_angles[i] and r > 1e-16:
                    gamma_angles[i] = r
            r = (bounds_angles[i][0] + np.pi - bounds_angles[i][-1])/2 #check extreme bounds (periodic)
            if r < gamma_angles[i] and r > 1e-16:
                gamma_angles[i] = r
        gamma_angles *= (1 - 1/max(h,w)) #guarantees that we have strict inequality    
        
        #construct matrices representing radius and angle value of each pixel
        radii = np.zeros([h,w])
        theta = np.zeros([h,w])
        h_center = h//2 + 1; w_center = w//2+1
        for i in range(0,h):
            for j in range(0,w):
                ri = (i+1.0 - h_center)*np.pi/h_center
                rj = (j+1.0 - w_center)*np.pi/w_center
                radii[i,j] = np.sqrt(ri**2 + rj**2)
                theta[i,j] = np.arctan2(ri,rj)
                if theta[i,j] <-.75*np.pi:
                    theta[i,j] += 2*np.pi
        
        mfb = []
        #construct scaling
        mfb.append([ewt2d_curveletScaling(radii,bounds_scales[0],gamma_scales)])
        
        #construct angular wedges for all but last scales
        for i in range(0,len(bounds_scales)-1):
            mfb_scale = []           
            for j in range(0,len(bounds_angles[i])-1):
                mfb_scale.append(ewt2d_curveletWavelet(theta,radii,
                                                       bounds_angles[i][j],bounds_angles[i][j+1],
                                                       bounds_scales[i],bounds_scales[i+1],
                                                       gamma_angles[i],gamma_scales))
            mfb_scale.append(ewt2d_curveletWavelet(theta,radii,
                                                       bounds_angles[i][-1],bounds_angles[i][0]+np.pi,
                                                       bounds_scales[i],bounds_scales[i+1],
                                                       gamma_angles[i],gamma_scales))
            mfb.append(mfb_scale)
        #construct angular wedges for last scales
    
        mfb_scale = []            
        for j in range(0,len(bounds_angles[-1])-1):
            mfb_scale.append(ewt2d_curveletWavelet(theta,radii,
                                                   bounds_angles[-1][j],bounds_angles[-1][j+1],
                                                   bounds_scales[-1],np.pi*2,
                                                   gamma_angles[-1],gamma_scales))
        mfb_scale.append(ewt2d_curveletWavelet(theta,radii,
                                                   bounds_angles[-1][-1],bounds_angles[-1][0]+np.pi,
                                                   bounds_scales[-1],np.pi*2,
                                                   gamma_angles[-1],gamma_scales))
        mfb.append(mfb_scale)
    elif option == 3:
        # Angles detected first then scales per angles
        
        #compute gamma for theta
        gamma_angles = 2*np.pi
        for k in range(0,len(bounds_angles)-1):
            r = (bounds_angles[k+1] - bounds_angles[k])/2
            if r < gamma_angles and r > 1e-16:
                gamma_angles = r
        r = (bounds_angles[0] + np.pi - bounds_angles[-1])/2 #check extreme bounds (periodic)
        if r < gamma_angles and r > 1e-16:
            gamma_angles = r
        gamma_angles *= (1 - 1/max(h,w)) #guarantees that we have strict inequality    
        
        #compute gamma for scales
        gamma_scales = bounds_scales[0][0]/2
        for i in range(1,len(bounds_angles)):
            for j in range(0,len(bounds_scales[i])-1):
                r = (bounds_scales[i][j+1] - bounds_scales[i][j])/(bounds_scales[i][j+1] + bounds_scales[i][j])
                if r < gamma_scales and r > 1e-16:
                    gamma_scales = r
            r = (np.pi-bounds_scales[i][-1])/(np.pi+bounds_scales[i][-1])
            if r < gamma_scales and r > 1e-16:
                gamma_scales = r
        gamma_scales *= (1-1/max(h,w))
        
        radii = np.zeros([h,w])
        theta = np.zeros([h,w])
        h_center = h//2 + 1; w_center = w//2+1
        for i in range(0,h):
            for j in range(0,w):
                ri = (i+1.0 - h_center)*np.pi/h_center
                rj = (j+1.0 - w_center)*np.pi/w_center
                radii[i,j] = np.sqrt(ri**2 + rj**2)
                theta[i,j] = np.arctan2(ri,rj)
        
        #Get empirical scaling function
        mfb = []
        mfb.append([ewt2d_curveletScaling(radii,bounds_scales[0][0],gamma_scales)])
        
        #for each angular sector, get empirical wavelet 
        for i in range(0,len(bounds_angles)-1):
            mfb_scale = []
            #generate first scale
            mfb_scale.append(ewt2d_curveletWavelet(theta,radii,
                                                   bounds_angles[i],bounds_angles[i+1],
                                                   bounds_scales[0][0],bounds_scales[i+1][0],
                                                   gamma_angles,gamma_scales))
            #generate for other scales
            for j in range(0,len(bounds_scales[i+1])-1):
                mfb_scale.append(ewt2d_curveletWavelet(theta,radii,
                                                   bounds_angles[i],bounds_angles[i+1],
                                                   bounds_scales[i+1][j],bounds_scales[i+1][j+1],
                                                   gamma_angles,gamma_scales))
            #generate for last scale
            mfb_scale.append(ewt2d_curveletWavelet(theta,radii,
                                                   bounds_angles[i],bounds_angles[i+1],
                                                   bounds_scales[i+1][-1],2*np.pi,
                                                   gamma_angles,gamma_scales))
            mfb.append(mfb_scale)
        
        #Generate for last angular sector
        mfb_scale = []
        mfb_scale.append(ewt2d_curveletWavelet(theta,radii,
                                                   bounds_angles[-1],bounds_angles[0]+np.pi,
                                                   bounds_scales[0][0],bounds_scales[-1][0],
                                                   gamma_angles,gamma_scales))
        for i in range(0,len(bounds_scales[-1])-1):
            mfb_scale.append(ewt2d_curveletWavelet(theta,radii,
                                                   bounds_angles[-1],bounds_angles[0]+np.pi,
                                                   bounds_scales[-1][i],bounds_scales[-1][i+1],
                                                   gamma_angles,gamma_scales))
        mfb_scale.append(ewt2d_curveletWavelet(theta,radii,
                                                   bounds_angles[-1],bounds_angles[0]+np.pi,
                                                   bounds_scales[-1][-1],2*np.pi,
                                                   gamma_angles,gamma_scales))
        mfb.append(mfb_scale)
    else:
        print('invalid option')
        return -1
    
    if h_extended == 1: #if we extended the height of the image, trim
        h -= 1
        for i in range(0,len(mfb)):
            for j in range(0,len(mfb[i])):
                mfb[i][j] = mfb[i][j][0:-1,:]
    if w_extended == 1: #if we extended the width of the image, trim
        w -= 1
        for i in range(0,len(mfb)):
            for j in range(0,len(mfb[i])):
                mfb[i][j] = mfb[i][j][:,0:-1]
    #invert the fftshift since filters are centered
    for i in range(0,len(mfb)):
        for j in range(0,len(mfb[i])):
            mfb[i][j] = np.fft.ifftshift(mfb[i][j])
    #resymmetrize for even images
    if option < 3:
        if h_extended == 1:
            s = np.zeros(w)
            if w%2 == 0:
                for j in range(0,len(mfb[-1])):
                    mfb[-1][j][h//2, 1:w//2] += mfb[-1][j][h//2, -1:w//2:-1]
                    mfb[-1][j][h//2, w//2+1:] = mfb[-1][j][h//2, w//2-1:0:-1]
                    s += mfb[-1][j][h//2,:]**2
                #normalize for tight frame
                for j in range(0,len(mfb[-1])):
                    mfb[-1][j][h//2, 1:w//2] /= np.sqrt(s[1:w//2])
                    mfb[-1][j][h//2, w//2+1:] /= np.sqrt(s[w//2+1:])
            else:
                for j in range(0,len(mfb[-1])):
                    mfb[-1][j][h//2, 0:w//2] += mfb[-1][j][h//2, -1:w//2:-1]
                    mfb[-1][j][h//2, w//2+1:] = mfb[-1][j][h//2, w//2-1::-1]
                    s += mfb[-1][j][h//2,:]**2
                for j in range(0,len(mfb[-1])):
                    mfb[-1][j][h//2,0:w//2]  /= np.sqrt(s[0:w//2])
                    mfb[-1][j][h//2,w//2+1:] /= np.sqrt(s[w//2+1:])
        if w_extended == 1:
            s = np.zeros(h)
            if h%2 == 0:
                for j in range(0,len(mfb[-1])):
                    mfb[-1][j][1:h//2, w//2] += mfb[-1][j][-1:h//2:-1, w//2]
                    mfb[-1][j][h//2+1:, w//2] = mfb[-1][j][h//2-1:0:-1, w//2]
                    s += mfb[-1][j][:, w//2]**2
                #normalize for tight frame
                for j in range(0,len(mfb[-1])):
                    mfb[-1][j][1:h//2, w//2] /= np.sqrt(s[1:h//2])
                    mfb[-1][j][h//2+1:, w//2] /= np.sqrt(s[h//2+1:]) 
            else:
                for j in range(0,len(mfb[-1])):
                    mfb[-1][j][0:h//2, w//2] += mfb[-1][j][-1:h//2:-1, w//2]
                    mfb[-1][j][h//2+1:, w//2] = mfb[-1][j][h//2-1::-1, w//2]
                    s += mfb[-1][j][:, w//2]**2
                for j in range(0,len(mfb[-1])):
                    mfb[-1][j][0:h//2, w//2] /= s[0:h//2]
                    mfb[-1][j][h//2+1:, w//2] /= s[h//2+1:]
    else:
        if h_extended == 1:
            s = np.zeros(w)
            if w%2 == 0:
                for j in range(0,len(mfb)):
                    mfb[j][-1][h//2, 1:w//2] += mfb[j][-1][h//2, -1:w//2:-1]
                    mfb[j][-1][h//2, w//2+1:] = mfb[j][-1][h//2, w//2-1:0:-1]
                    s += mfb[j][-1][h//2,:]**2
                #normalize for tight frame
                for j in range(0,len(mfb)):
                    mfb[j][-1][h//2, 1:w//2] /= np.sqrt(s[1:w//2])
                    mfb[j][-1][h//2, w//2+1:] /= np.sqrt(s[w//2+1:])
            else:
                for j in range(0,len(mfb)):
                    mfb[j][-1][h//2, 0:w//2] += mfb[j][-1][h//2, -1:w//2:-1]
                    mfb[j][-1][h//2, w//2+1:] = mfb[j][-1][h//2, w//2-1::-1]
                    s += mfb[j][-1][h//2,:]**2
                for j in range(0,len(mfb)):
                    mfb[j][-1][h//2,0:w//2]  /= np.sqrt(s[0:w//2])
                    mfb[j][-1][h//2,w//2+1:] /= np.sqrt(s[w//2+1:])
        if w_extended == 1:
            s = np.zeros(h)
            if h%2 == 0:
                for j in range(0,len(mfb)):
                    mfb[j][-1][1:h//2, w//2] += mfb[j][-1][-1:h//2:-1, w//2]
                    mfb[j][-1][h//2+1:, w//2] = mfb[j][-1][h//2-1:0:-1, w//2]
                    s += mfb[j][-1][:, w//2]**2
                #normalize for tight frame
                for j in range(0,len(mfb)):
                    mfb[j][-1][1:h//2, w//2] /= np.sqrt(s[1:h//2])
                    mfb[j][-1][h//2+1:, w//2] /= np.sqrt(s[h//2+1:]) 
            else:
                for j in range(0,len(mfb)):
                    mfb[j][-1][0:h//2, w//2] += mfb[j][-1][-1:h//2:-1, w//2]
                    mfb[j][-1][h//2+1:, w//2] = mfb[j][-1][h//2-1::-1, w//2]
                    s += mfb[j][-1][:, w//2]**2
                for j in range(0,len(mfb)):
                    mfb[j][-1][0:h//2, w//2] /= s[0:h//2]
                    mfb[j][-1][h//2+1:, w//2] /= s[h//2+1:]
    return mfb

"""
ewt2d_curveletScaling(radii,bound,gamma)
Constructs the empirical Curvelet scaling function (circle)
Input:
    radii   - reference image where pixels are equal to their distance from 
            center
    bound   - first radial bound
    gamma   - detected scale gamma to guarantee tight frame
Output:
    scaling - resulting empirical Curvelet scaling function
Author: Basile Hurat, Jerome Gilles""" 
def ewt2d_curveletScaling(radii, bound, gamma):
    
    an = 1/(2*gamma*bound) 
    mbn = (1 - gamma)*bound # inner circle up to beginning of transtion
    pbn = (1 + gamma)*bound #end of transition
    scaling = 0*radii #initiate w/ zeros
    scaling[radii < mbn] = 1
    scaling[radii >= mbn] = np.cos(np.pi*ewt_beta(an*(radii[radii>=mbn] - mbn))/2)
    scaling[radii > pbn] = 0
    return scaling

"""
ewt2d_curveletWavelet(theta, radii,ang_bound1,ang_bound2, scale_bound1, 
                      scale_bound2,gamma_angle,gamma_scale)
Constructs the empirical Curvelet wavelet function (polar wedge)
Input:
    theta       - reference image where pixels are equal to their angle from center
    radii       - reference image where pixels are equal to their distance from 
                center
    ang_bound1  - lower angular bound
    ang_bound2  - upper angular bound
    scale_bound1- lower radial bound
    scale_bound2- upper radial bound
    gamma_angle - detected angle gamma to guarantee tight frame
    gamma_scale - detected scale gamma to guarantee tight frame
Output:
    wavelet - resulting empirical Curvelet wavelet
Author: Basile Hurat, Jerome Gilles""" 
def ewt2d_curveletWavelet(theta, radii, ang_bound1, ang_bound2, scale_bound1, scale_bound2, gamma_angle, gamma_scale):
    #radial parameters
    wan = 1/(2*gamma_scale*scale_bound1) #scaling factor
    wam = 1/(2*gamma_scale*scale_bound2) 
    wmbn = (1 - gamma_scale)*scale_bound1 #beginning of lower transition
    wpbn = (1 + gamma_scale)*scale_bound1 #end of lower transition
    wmbm = (1 - gamma_scale)*scale_bound2  #beginning of upper transition
    wpbm = (1 + gamma_scale)*scale_bound2 #end of upper transition
    
    #angular parameters 
    an = 1/(2*gamma_angle)
    mbn = ang_bound1 - gamma_angle
    pbn = ang_bound1 + gamma_angle
    mbm = ang_bound2 - gamma_angle
    pbm = ang_bound2 + gamma_angle
    
    wavelet = 0*theta #initialize w/ zeros
    if ang_bound2 - ang_bound1 != np.pi:
        inside = (theta >= mbn)*(theta < pbm) #
    else:
        inside = (theta >= ang_bound1)*(theta <= ang_bound2)
    inside *= (radii > wmbn)*(radii < wpbm)
    wavelet[inside] = 1.0 #set entire angular wedge equal to 1
    temp = inside*(radii >= wmbm)*(radii <= wpbm) #upper radial transition
    wavelet[temp] *= np.cos(np.pi*ewt_beta(wam*(radii[temp]-wmbm))/2)
    temp = inside*(radii >= wmbn)*(radii <= wpbn) #lower radial transition
    wavelet[temp] *= np.sin(np.pi*ewt_beta(wan*(radii[temp]-wmbn))/2)
    
    if ang_bound2 - ang_bound1 != np.pi:
        temp = inside*(theta >= mbm)*(theta <= pbm) #upper angular transition
        wavelet[temp] *= np.cos(np.pi*ewt_beta(an*(theta[temp]-mbm))/2)
        temp = inside*(theta >= mbn)*(theta <= pbn) #lower angular transition
        wavelet[temp] *= np.sin(np.pi*ewt_beta(an*(theta[temp]-mbn))/2)

    return wavelet + wavelet[-1::-1,-1::-1] #symmetrize
    
"""
ppfft(f)
Performs the pseudo-polar fast Fourier transform of image f
Input:
    f       - input image f
Output:
    ppff    - pseudo-polar fourier transform of image f
Author: Basile Hurat""" 
def ppfft(f):
    #f is assumed N x N where N is even. If not, we just force it to be
    [h,w] = f.shape
    N = h
    f2 = f
    if h != w or np.mod(h,2) == 1:
        N = int(np.ceil(max(h,w)/2)*2) #N is biggest dimension, but force even
        f2 = np.zeros([N,N])
        f2[N//2-int(h/2):N//2-int(h/2)+h,N//2-int(w/2):N//2-int(w/2)+w] = f
    ppff = np.zeros([2*N,2*N])*1j
    
    #Constructing Quadrants 1 and 3
    ff = np.fft.fft(f2,N*2,axis = 0)
    ff = np.fft.fftshift(ff,0)
    for i in range(-N,N):
        ppff[i+N,N-1::-1] = fracfft(ff[i+N,:],i/(N**2),1)
    
    #Constructing quadrants 2 and 4
    ff = np.fft.fft(f2,N*2,axis = 1)
    ff = np.fft.fftshift(ff,1)
    ff = ff.T
    
    for i in range(-N,N):
        x = np.arange(0,N)
        factor = np.exp(1j*2*np.pi*x*(N/2-1)*i/(N**2))
        ppff[i+N,N:2*N] = fracfft(ff[i+N,:]*factor,i/(N**2))
    return ppff

"""
fracfft(f)
Performs the fractional fast Fourier transform of image f
Input:
    f           - input image f
    alpha       - fractional value for fractional fft
    centered    - whether or not this is centered
Output:
    result      - fractional fourier transform of image f
Author: Basile Hurat""" 
def fracfft(f,alpha,centered = 0):
    f = np.reshape(f.T,f.size)#flatten f
    N = len(f)#get length
    
    if centered == 1:
        x = np.arange(0,N)
        factor = np.exp(1j*np.pi*x*N*alpha)
        f = f*factor
    
    x = np.append(np.arange(0,N),np.arange(-N,0))
    factor = np.exp(-1j*np.pi*alpha*x**2)
    ff = np.append(f,np.zeros(N))
    ff = ff*factor
    XX = np.fft.fft(ff)
    YY = np.fft.fft(np.conj(factor))
    
    result = np.fft.ifft(XX*YY)
    result = result*factor
    result = result[0:N]
    return result

def ippfft(ppff,acc = 1e-5):
    [h,w] = ppff.shape
    h = h//2
    
    w = np.sqrt(np.abs(np.arange(-h,h))/2)/h
    w[h+1] = np.sqrt(1/8)/h
    w = np.outer(np.ones(2*h),w)
    recon = np.zeros([h,h])
    delta = 1
    count = 0
    while delta > acc and count < 1000:
        error = w*(ppfft(recon).T-ppff.T)
        D=appfft(w*error).T
        delta=np.linalg.norm(D)
        mu=1/h
        recon=recon-mu*D
        count += 1
    if count == 1000:
        print('could not converge during inverse pseudo-polar fft')
    return recon

def appfft(X):
    [h,w] = X.shape
    h = h//2
    
    Y = 1j*np.zeros([h,h])
    temp = 1j*np.zeros([h,2*h])
    for i in range(-h,h):
        Xvec = X[h-1::-1,i+h]
        alpha = -i/(h**2)
        OneLine = fracfft(Xvec,alpha)
        OneLine = (OneLine.T)*np.exp(-1j*np.pi*i*np.arange(0,h)/h)
        temp[:,i+h] = OneLine
    Temp_Array = 2*h*np.fft.ifft(temp,axis = 1)
    Temp_Array = Temp_Array[:,0:h].dot(np.diag(np.power(-1,np.arange(0,h))))
    Y = Temp_Array.T
    
    temp2 = 1j*np.zeros([h,2*h])
    for i in range(-h,h):
        Xvec = X[-1:h-1:-1,i+h]
        alpha = i/(h**2)
        OneCol = fracfft(Xvec,alpha)
        OneCol = (OneCol)*(np.exp(1j*np.pi*i*np.arange(0,h)/h).T)
        temp2[:,i+h] = OneCol
    Temp_Array = 2*h*np.fft.ifft(temp2,axis = 1)
    Y += Temp_Array[:,0:h].dot(np.diag(np.power(-1,np.arange(0,h))))
    Y = Y.T
    return Y