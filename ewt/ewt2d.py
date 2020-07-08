# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 13:59:13 2020

@author: bazzz
"""
import numpy as np
from ewt.boundaries import *
from ewt.ewt1d import *
def ewt2dTensor(f,params):
    ff = np.fft.fft(f.T) #take 1D fft along columns
    meanfft = np.mean(np.abs(ff),axis = 0) #take average of magnitude
    bounds_col = ewt_boundariesDetect(meanfft[0:int(np.round(len(meanfft)/2))],params)
    bounds_col = bounds_col*2*np.pi/len(meanfft)
    ewtc_col = []
    mfb_col = EWT_LP_Filterbank(bounds_col,len(meanfft)) #construct filter bank
    for i in range(0,len(mfb_col[0,:])):
        filter_col = np.tile(mfb_col[:,i],[len(f[0,:]),1]) #repeat down
        ewtc_col.append(np.real(np.fft.ifft(filter_col*ff)).T) #get coefficients,
    
    # REPEAT WITH ROWS #
    ff = np.fft.fft(f) #take 1D fft along rows
    meanfft = np.mean(np.abs(ff),axis = 0) 
    bounds_row = ewt_boundariesDetect(meanfft[0:int(np.round(len(meanfft)/2))],params)
    bounds_row = bounds_row*2*np.pi/len(meanfft)

    ewtc = []
    mfb_row = EWT_LP_Filterbank(bounds_row,len(meanfft))
    for i in range(0,len(mfb_row[0,:])):
        ewtc_row = []
        filter_row = np.tile(mfb_row[:,i],[len(f[:,0]),1])
        for j in range(0,len(mfb_col[0,:])):
            ff = np.fft.fft(ewtc_col[j])
            ewtc_row.append(np.real(np.fft.ifft(filter_row*ff)))
        ewtc.append(ewtc_row)
            
    return [ewtc, mfb_row, mfb_col,bounds_row,bounds_col]
    
def iewt2dTensor(ewtc,mfb_row,mfb_col):
    [h,w]= ewtc[0][0].shape
    ewt_col = []
    for i in range(0,len(mfb_col[0,:])):
        for j in range(0,len(mfb_row[0,:])):
            ff = np.fft.fft(ewtc[j][i])
            filter_row = np.tile(mfb_row[:,j],[h, 1])
            if j == 0:
                ewt_col.append(np.zeros([h,w]))
            ewt_col[i] += np.real(np.fft.ifft(filter_row*ff))
    
    for i in range(0,len(mfb_col[0,:])):
        ff = np.fft.fft(ewt_col[i].T);
        if i == 0:
            img = np.zeros([h,w])
        
        filter_col = np.tile(mfb_col[:,i],[w, 1])
        img = img + np.real(np.fft.ifft(filter_col*ff)).T
    return img

def ewt2dCurvelet(f,params):
    [h,w] = f.shape
    ppff = ppfft(f)
    
    if params.option == 1:
    #Option 1: Computes scales and angles independently    
        #Begin with scales
        meanppff = np.fft.fftshift(np.mean(np.abs(ppff),axis = 1))
        bounds_scales = ewt_boundariesDetect(meanppff[0:len(meanppff)//2+1],params)
        bounds_scales *= np.pi/np.ceil((len(meanppff)/2))
        
        #Then do with angles
        meanppff = np.mean(np.abs(ppff),axis = 0)
        bounds_angles = ewt_boundariesDetect(meanppff[0:len(meanppff)//2+1],params)
        bounds_angles = bounds_angles*np.pi/np.ceil((len(meanppff)/2)) - np.pi*.75
    
    elif params.option == 2:
        #Option 2: Computes Scales first, then angles 
        meanppff = np.fft.fftshift(np.mean(np.abs(ppff),axis = 1))
        bounds_scales = ewt_boundariesDetect(meanppff[0:len(meanppff)//2+1],params)
        bounds_angles = []
        for i in range(0,len(bounds_scales)-1):
            meanppff = np.mean(np.abs(ppff[int(bounds_scales[i]):int(bounds_scales[i+1]+1),:]),axis = 0)
            bounds =  ewt_boundariesDetect(meanppff[0:len(meanppff)//2+1],params)
            #append
            bounds_angles.append(bounds*np.pi/np.ceil((len(meanppff)/2)) - np.pi*.75)
            
        #Do last linterval
        meanppff = np.mean(np.abs(ppff[int(bounds_scales[-1]):,:]),axis = 0)
        bounds =  ewt_boundariesDetect(meanppff[0:len(meanppff)//2+1],params)
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
        bound0 = ewt_boundariesDetect(meanppff[LL:],params)[0]
        bounds_scales.append([bound0*np.pi/LL])
        bound0 = int(bound0)
        #Compute mean-pseudo-polar fft for angles, excluding first scale to 
        #find angle bounds
        meanppff = np.mean(np.abs(ppff[ppff.shape[0]//2+bound0:,:]),0)
        bounds_theta = ewt_boundariesDetect(meanppff,params)
        bounds_angles = (bounds_theta-1)*np.pi/len(meanppff)-0.75*np.pi
        bounds_theta = bounds_theta.astype(int)
        #Now we find scale bounds at each angle
        for i in range(0,len(bounds_theta)-1):
            meanppff = np.mean(np.abs(ppff[LL+bound0:,bounds_theta[i]:bounds_theta[i+1]+1]),1)
            bounds = ewt_boundariesDetect(meanppff,params)
            bounds_scales.append((bounds+bound0)*np.pi/LL)
        
        #and also for the last angle
        meanppff = np.mean(np.abs(ppff[LL+bound0:,bounds_theta[-1]:]),1)
        meanppff += np.mean(np.abs(ppff[LL+bound0:,1:bounds_theta[0]+1]),1)
        params.spectrumRegularize = 'closing'
        bounds = ewt_boundariesDetect(meanppff,params)
        bounds_scales.append((bounds+bound0)*np.pi/LL)
    else:
        print('invalid option')
        return -1
    #Once bounds are found, construct filter bank, take fourier transform of 
    #image, and filter
    mfb = curveletFilterbank(bounds_scales,bounds_angles,h,w,params.option)
    ff = np.fft.fft2(f)
    ###ewtc = result!
    ewtc = []
    for i in range(0,len(mfb)):
        ewtc_scales = []
        for j in range(0,len(mfb[i])):
            ewtc_scales.append(np.real(np.fft.ifft2(mfb[i][j]*ff)))
        ewtc.append(ewtc_scales)
    return [ewtc, mfb, bounds_scales, bounds_angles]

def iewt2dCurvelet(ewtc,mfb):
    recon = np.fft.fft2(ewtc[0][0])*mfb[0][0]
    for i in range(1,len(mfb)):
        for j in range(0,len(mfb[i])):
            recon += np.fft.fft2(ewtc[i][j])*mfb[i][j]
    return np.real(np.fft.ifft2(recon))
            
def curveletFilterbank(bounds_scales,bounds_angles,h,w,option):

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
                                                   bounds_angles[-1]-np.pi,bounds_angles[0],
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

def ewt2d_curveletScaling(radii, bound, gamma):
    
    an = 1/(2*gamma*bound) 
    mbn = (1 - gamma)*bound # inner circle up to beginning of transtion
    pbn = (1 + gamma)*bound #end of transition
    scaling = 0*radii #initiate w/ zeros
    scaling[radii < mbn] = 1
    scaling[radii >= mbn] = np.cos(np.pi*EWT_beta(an*(radii[radii>=mbn] - mbn))/2)
    scaling[radii > pbn] = 0
    return scaling

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
    inside = (theta > mbn)*(theta < pbm) #
    inside *= (radii > wmbn)*(radii < wpbm)
    wavelet[inside] = 1.0 #set entire angular wedge equal to 1
    temp = inside*(radii >= wmbm)*(radii <= wpbm) #upper radial transition
    wavelet[temp] *= np.cos(np.pi*EWT_beta(wam*(radii[temp]-wmbm))/2)
    temp = inside*(radii >= wmbn)*(radii <= wpbn) #lower radial transition
    wavelet[temp] *= np.sin(np.pi*EWT_beta(wan*(radii[temp]-wmbn))/2)
    temp = inside*(theta >= mbm)*(theta <= pbm) #upper angular transition
    wavelet[temp] *= np.cos(np.pi*EWT_beta(an*(theta[temp]-mbm))/2)
    temp = inside*(theta >= mbn)*(theta <= pbn) #lower angular transition
    wavelet[temp] *= np.sin(np.pi*EWT_beta(an*(theta[temp]-mbn))/2)
    
    return wavelet + wavelet[-1::-1,-1::-1] #symmetrize
    

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


    
    
    
    