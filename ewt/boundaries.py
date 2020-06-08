# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:15:58 2020

@author: bazzz
"""
import numpy as np
from scipy.special import iv

def ewt_boundariesDetect(absf):
    plane = GSS(absf)
    [lengths, indices] = lengthScaleCurve(plane)
    thresh = otsu(lengths)
    bounds = indices[lengths >= thresh]
    return bounds

def GSS(f):
    t = 0.5
    n = 3
    num_iter = np.ceil(len(f)/n)
    #First, define scale-space kernel (discrete Gaussian kernel)
    ker = np.exp(-t)*iv(np.arange(-n,n+1),t)
    ker = ker/np.sum(ker)

    #Initialize place to store result of each layer GSS
    plane = np.zeros([len(f), num_iter.astype(int)+1])
    plane[:,0] = localmin(f)

    #Iterate through scalespace and store minima at each scale
    for i in range(1,num_iter.astype(int)+1):
        f = np.pad(f,n,'reflect')
        f = np.convolve(f,ker,'same')
        f = f[n:-n]
        plane[:,i] = localmin(f)
    return plane

def lengthScaleCurve(plane):
    [w,num_iter] = plane.shape
    num_curves = np.sum(plane[:,0])
    lengths = np.ones(num_curves.astype(int))
    indices = np.zeros(num_curves.astype(int))
    current_curve = 0;
 
    for i in range(0,w):
        if plane[i,0] == 1:
            indices[current_curve] = i
            i0 = i
            height = 2
            stop = 0
            while stop == 0:
                flag = 0
                for p in range(-1,2):
                    if (i+p  < 0) | (i + p >= w):
                        continue
                    #If minimum at next iteration of scale-space, increment length
                    #height, minimum location
                    if plane[i + p,height] == 1: 
                        lengths[current_curve] += 1 
                        height += 1
                        i0 += p
                        flag = 1
                        #Stop if pas number of iterations
                        if height >= num_iter:  
                            stop = 1
                        break
                #Stop if no minimum found
                if flag == 0:
                    stop = 1
            #Go to next curve/minimum after done
            current_curve += 1  

    return [lengths, indices]

def localmin(f):
    w = len(f)
    minima = np.zeros(w)
    for i in range(0,w):
        minima[i] = 1
        right = 1
        while 1:
            if i - right >= 0:
                if f[i - right] < f[i]:
                    minima[i] = 0
                    break
                elif f[i - right] == f[i]:
                    right += 1
                else:
                    break
            else:
                break
        if minima[i] == 1:
            left = 1
            while 1:
                if i + left < w:
                    if f[i+left] < f[i]:
                        minima[i] = 0
                        break
                    elif f[i+left] == f[i]:
                        left += 1
                    else:
                        break
                else:
                    break
    i = 0
    while i < w:
        if minima[i] == 1:
            j = i
            flat_count = 1
            flag = 0
            while (j+1 < w) and (minima[j+1] == 1):
                minima[j] = 0
                flat_count += 1
                j += 1
                flag = 1
                print(minima)
            if flag == 1:
                minima[j - np.floor(flat_count/2).astype(int)] = 1
                minima[j] = 0
                i = j
        i += 1
                
    return minima

def otsu(lengths):
    hist_max = np.max(lengths); 
    histogram = np.histogram(lengths,hist_max.astype(int))[0]
    hist_normalized = histogram/np.sum(histogram) #normalize
    Q = hist_normalized.cumsum()
    
    bins = np.arange(hist_max)
    fn_min = np.inf
    thresh = -1

    for i in range(1,hist_max.astype(int)):
        p1, p2 = np.hsplit(hist_normalized, [i])
        q1, q2 = Q[i], Q[hist_max.astype(int)-1] - Q[i]
        if q1 < 1.e-6 or q2 < 1.e-6:
            continue
        b1, b2 = np.hsplit(bins,[i]) #weights

        #Means and variances
        m1 = np.sum(p1*b1)/q1
        m2 = np.sum(p2*b2)/q2
        v1 = np.sum((b1-m1)**2*p1)/q1
        v2 = np.sum((b2-m2)**2*p2)/q2

        #calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    return thresh