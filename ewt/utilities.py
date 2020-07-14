import numpy as np
from scipy.signal import gaussian
import matplotlib.pyplot as plt
from matplotlib import patches

"""
ewt_params()
Parameter struct for empirical wavelet. Also sets defaults for each value.
Parameters are as follow:
    log                 - whether or not you take log of spectrum
    removeTrends        - Removes trends before finding boundaries
    spectrumRegularize  - regularizes spectrum before finding boundaries
    lengthFilter        - length for filters used in spectrum regularization
    sigmaFilter         - sigma for gaussian filter used in spectrum regularization
    typeDetect          - type of thresholding method for scale-space detection
    option              - Curvelet option for 2D Curvelet EWT
Author: Basile Hurat, Jerome Gilles"""
class ewt_params:
    def __init__(self):
        self.log = 0
        self.removeTrends = 'none'
        self.spectrumRegularize = 'none'
        self.lengthFilter = 7
        self.sigmaFilter = 2
        self.typeDetect = 'otsu'
        self.option = 1

"""
spectrumRegularize(f,params)
pre-processes spectrum before boundary detection by regularizing spectrum
Options include:
    gaussian    - gaussian smoothing: lengthFilter defines size of filter and 
                sigmaFilter defines std dev of gaussian
    average     - box filter smoothing: lengthFilter defines size of filter
    closing     - compute the upper envelope via a morphological closing 
                operator: lengthFilter defines size of filter
Input:
    f       - spectrum to regularize
    params  - parameters for EWT (see utilities) Used for choice of 
            regularization and details
Output:
    f2      - regularized spectrum
Author: Basile Hurat, Jerome Gilles"""
def spectrumRegularize(f, params):
    if params.spectrumRegularize.lower() == 'gaussian': #gaussian
        f2 = np.pad(f,params.lengthFilter//2,'reflect')
        Reg_Filter = gaussian(params.lengthFilter,params.sigmaFilter)
        Reg_Filter = Reg_Filter/sum(Reg_Filter)
        f2 = np.convolve(f2,Reg_Filter,mode = 'same')
        return f2[params.lengthFilter//2:-params.lengthFilter//2]
    elif params.spectrumRegularize.lower() == 'average': #average
        f2 = np.pad(f,params.lengthFilter//2,'reflect')
        Reg_Filter = np.ones(params.lengthFilter)
        Reg_Filter = Reg_Filter/sum(Reg_Filter)
        f2 = np.convolve(f2,Reg_Filter,mode = 'same')
        return f2[params.lengthFilter//2:-params.lengthFilter//2]
    elif params.spectrumRegularize.lower() == 'closing': #closing
        f2 = np.zeros(len(f))
        for i in range(0,len(f)):
            f2[i] = np.min(f[max(0 , i - params.lengthFilter) : min(len(f)-1, i + params.lengthFilter+1)])
        return f2
    
def removeTrends(f, params):
    #still needs to be implemented
    return f

"""
showewt1dBoundaries(f,bounds)
Plots boundaries of 1D EWT on top of magnitude spectrum of the signal
Input:
    f       - original signal
    bounds  - detected bounds
Author: Basile Hurat, Jerome Gilles"""
def showewt1dBoundaries(f,bounds):
    ff = np.abs(np.fft.fft(f))
    h = np.max(ff)
    plt.figure()
    plt.plot(np.arange(0,np.pi+1/(len(ff)/2),np.pi/(len(ff)/2)),ff[0:len(ff)//2+1])
    for i in range(0,len(bounds)):
        plt.plot([bounds[i], bounds[i]],[0,h-1],'r--')
    plt.show()
"""
showTensorBoundaries(f,bounds_row,bounds_col)
Plots boundaries of 2D tensor EWT on top of magnitude spectrum of image
Input:
    f           - original image
    bounds_row  - detected bounds on rows
    bounds_col  - detected bounds on columns
Author: Basile Hurat, Jerome Gilles"""
def show2DTensorBoundaries(f,bounds_row,bounds_col):
    [h,w] = f.shape
    ff = np.fft.fft2(f)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.imshow(np.log(np.abs(np.fft.fftshift(ff))),cmap = 'gray')
    
    #draw horizontal lines
    for i in range(0,len(bounds_row)): 
        plt.plot([h//2 + bounds_row[i]*h/np.pi/2,h//2 + bounds_row[i]*h/np.pi/2],[0,w-1],'r-')
        plt.plot([h//2 - bounds_row[i]*h/np.pi/2,h//2 - bounds_row[i]*h/np.pi/2],[0,w-1],'r-')
    #draw vertical lines
    for i in range(0,len(bounds_col)):
        plt.plot([0,h-1],[w//2 + bounds_col[i]*w/np.pi/2,w//2 + bounds_col[i]*w/np.pi/2],'r-')
        plt.plot([0,h-1],[w//2 - bounds_col[i]*w/np.pi/2, w//2 - bounds_col[i]*w/np.pi/2],'r-')
    plt.show()

def show2DLPBoundaries(f,bounds_scales):
    [h,w] = f.shape
    ff = np.fft.fft2(f)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.imshow(np.log(np.abs(np.fft.fftshift(ff))),cmap = 'gray')
    
    #plot scale bounds
    for i in range(0,len(bounds_scales)):
        rad = bounds_scales[i]*h/np.pi/2
        circ = plt.Circle((h//2+1,w//2+1),rad,color = 'r', Fill = 0)
        ax.add_patch(circ)
    plt.show()
"""
showCurveletBoundaries(f,option,bounds_scales,bounds_angles)
Plots boundaries of 2D curvelet EWT on top of magnitude spectrum of image
Input:
    f           - original image
    option      - option for Curvelet
    bounds_row  - detected bounds on scales
    bounds_col  - detected bounds on angles
Author: Basile Hurat, Jerome Gilles"""
def show2DCurveletBoundaries(f,option,bounds_scales,bounds_angles):
    [h,w] = f.shape
    ff = np.fft.fft2(f)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.imshow(np.log(np.abs(np.fft.fftshift(ff))),cmap = 'gray')
    if option == 1: #scales and angles detected separately
        #first plot scale bounds
        for i in range(0,len(bounds_scales)):
            rad = bounds_scales[i]*h/np.pi/2
            circ = plt.Circle((h//2+1,w//2+1),rad,color = 'r', Fill = 0)
            ax.add_patch(circ)
        #Then plot the angle bounds
        for i in range(0,len(bounds_angles)): 
            if abs(bounds_angles[i]) < np.pi/4: 
                #Do first half of line
                x0 = (1+bounds_scales[0]*np.cos(bounds_angles[i])/np.pi)*w//2
                y0 = (1+bounds_scales[0]*np.sin(bounds_angles[i])/np.pi)*h//2
                x1 = w-1
                y1 = (h+w*np.tan(bounds_angles[i]))//2
                plt.plot([x0,x1],[y0,y1],'r-')
                #Do second half of line
                x2 = (1-bounds_scales[0]*np.cos(bounds_angles[i])/np.pi)*w//2
                y2 = (1-bounds_scales[0]*np.sin(bounds_angles[i])/np.pi)*h//2
                x3 = 0
                y3 = (h-w*np.tan(bounds_angles[i]))//2
                plt.plot([x2,x3],[y2,y3],'r-')
            else:
                x0 = (1-bounds_scales[0]*np.cos(bounds_angles[i])/np.pi)*w//2
                y0 = (1-bounds_scales[0]*np.sin(bounds_angles[i])/np.pi)*h//2
                x1 = (w+h/np.tan(bounds_angles[i]))//2
                y1 = h-1
                plt.plot([x0,x1],[y0,y1],'r-')
                x2 = (1+bounds_scales[0]*np.cos(bounds_angles[i])/np.pi)*w//2
                y2 = (1+bounds_scales[0]*np.sin(bounds_angles[i])/np.pi)*h//2
                x3 = (h-w/np.tan(bounds_angles[i]))//2
                y3 = 0
                plt.plot([x2,x3],[y2,y3],'r-')
                
                
    elif option == 2: #scales detected first and angles detected per scale
        #first plot scale bounds
        for i in range(0,len(bounds_scales)):
            rad = bounds_scales[i]*h/np.pi/2
            circ = plt.Circle((h//2+1,w//2+1),rad,color = 'r', Fill = 0)
            ax.add_patch(circ)
        #Then plot the angle bounds for each scale
        for i in range(0,len(bounds_scales)-1):
            for j in range(0,len(bounds_angles[i])): 
                if abs(bounds_angles[i][j]) < np.pi/4: 
                    #Do first half of line
                    x0 = (1+bounds_scales[i]*np.cos(bounds_angles[i][j])/np.pi)*(w//2+1)
                    y0 = (1+bounds_scales[i]*np.sin(bounds_angles[i][j])/np.pi)*(h//2+1)
                    x1 = (1+bounds_scales[i+1]*np.cos(bounds_angles[i][j])/np.pi)*(w//2+1)
                    y1 = (1+bounds_scales[i+1]*np.sin(bounds_angles[i][j])/np.pi)*(h//2+1)
                    plt.plot([x0,x1],[y0,y1],'r-')
                    #Do second half of line
                    x2 = (1-bounds_scales[i]*np.cos(bounds_angles[i][j])/np.pi)*(w//2+1)
                    y2 = (1-bounds_scales[i]*np.sin(bounds_angles[i][j])/np.pi)*(h//2+1)
                    x3 = (1-bounds_scales[i+1]*np.cos(bounds_angles[i][j])/np.pi)*(w//2+1)
                    y3 = (1-bounds_scales[i+1]*np.sin(bounds_angles[i][j])/np.pi)*(h//2+1)
                    plt.plot([x2,x3],[y2,y3],'r-')
                else:
                    x0 = (1-bounds_scales[i]*np.cos(bounds_angles[i][j])/np.pi)*(w//2+1)
                    y0 = (1-bounds_scales[i]*np.sin(bounds_angles[i][j])/np.pi)*(h//2+1)
                    x1 = (1-bounds_scales[i+1]*np.cos(bounds_angles[i][j])/np.pi)*(w//2+1)
                    y1 = (1-bounds_scales[i+1]*np.sin(bounds_angles[i][j])/np.pi)*(h//2+1)
                    plt.plot([x0,x1],[y0,y1],'r-')
    
                    x2 = (1+bounds_scales[i]*np.cos(bounds_angles[i][j])/np.pi)*(w//2+1)
                    y2 = (1+bounds_scales[i]*np.sin(bounds_angles[i][j])/np.pi)*(h//2+1)
                    x3 = (1+bounds_scales[i+1]*np.cos(bounds_angles[i][j])/np.pi)*(w//2+1)
                    y3 = (1+bounds_scales[i+1]*np.sin(bounds_angles[i][j])/np.pi)*(h//2+1)
                    plt.plot([x2,x3],[y2,y3],'r-')
        #Then take care of last scale
        for i in range(0,len(bounds_angles[-1])): 
            if abs(bounds_angles[-1][i]) < np.pi/4: 
                #Do first half of line
                x0 = (1+bounds_scales[-1]*np.cos(bounds_angles[-1][i])/np.pi)*(w//2+1)
                y0 = (1+bounds_scales[-1]*np.sin(bounds_angles[-1][i])/np.pi)*(h//2+1)
                x1 = w-1
                y1 = (h+w*np.tan(bounds_angles[-1][i]))//2
                plt.plot([x0,x1],[y0,y1],'r-')
                #Do second half of line
                x2 = (1-bounds_scales[-1]*np.cos(bounds_angles[-1][i])/np.pi)*(w//2+1)
                y2 = (1-bounds_scales[-1]*np.sin(bounds_angles[-1][i])/np.pi)*(h//2+1)
                x3 = 0
                y3 = (h-w*np.tan(bounds_angles[-1][i]))//2
                plt.plot([x2,x3],[y2,y3],'r-')
            else:
                x0 = (1-bounds_scales[-1]*np.cos(bounds_angles[-1][i])/np.pi)*(w//2+1)
                y0 = (1-bounds_scales[-1]*np.sin(bounds_angles[-1][i])/np.pi)*(h//2+1)
                x1 = (w+h/np.tan(bounds_angles[-1][i]))//2
                y1 = h-1
                plt.plot([x0,x1],[y0,y1],'r-')
                x2 = (1+bounds_scales[-1]*np.cos(bounds_angles[-1][i])/np.pi)*(w//2+1)
                y2 = (1+bounds_scales[-1]*np.sin(bounds_angles[-1][i])/np.pi)*(h//2+1)
                x3 = (h-w/np.tan(bounds_angles[-1][i]))//2
                y3 = 0
                plt.plot([x2,x3],[y2,y3],'r-')
    elif option == 3: #angles detected first and scales detected per angle
        #plot first scale
        rad = bounds_scales[0][0]*h/np.pi/2
        circ = plt.Circle((h//2,w//2),rad,color = 'r', Fill = 0)
        ax.add_patch(circ)
        
        #Plot angle bounds first
        for i in range(0,len(bounds_angles)): 
            if abs(bounds_angles[i]) < np.pi/4: 
                #Do first half of line
                x0 = (1+bounds_scales[0][0]*np.cos(bounds_angles[i])/np.pi)*w//2
                y0 = (1+bounds_scales[0][0]*np.sin(bounds_angles[i])/np.pi)*h//2
                x1 = w-1
                y1 = (h+w*np.tan(bounds_angles[i]))//2
                plt.plot([x0,x1],[y0,y1],'r-')
                #Do second half of line
                x2 = (1-bounds_scales[0][0]*np.cos(bounds_angles[i])/np.pi)*w//2
                y2 = (1-bounds_scales[0][0]*np.sin(bounds_angles[i])/np.pi)*h//2
                x3 = 0
                y3 = (h-w*np.tan(bounds_angles[i]))//2
                plt.plot([x2,x3],[y2,y3],'r-')
            else:
                x0 = (1-bounds_scales[0][0]*np.cos(bounds_angles[i])/np.pi)*w//2
                y0 = (1-bounds_scales[0][0]*np.sin(bounds_angles[i])/np.pi)*h//2
                x1 = (w+h/np.tan(bounds_angles[i]))//2
                y1 = h-1
                plt.plot([x0,x1],[y0,y1],'r-')
                x2 = (1+bounds_scales[0][0]*np.cos(bounds_angles[i])/np.pi)*w//2
                y2 = (1+bounds_scales[0][0]*np.sin(bounds_angles[i])/np.pi)*h//2
                x3 = (h-w/np.tan(bounds_angles[i]))//2
                y3 = 0
                plt.plot([x2,x3],[y2,y3],'r-')
        #For each angular sector, plot arc for scale
        for i in range(0,len(bounds_angles)-1): 
            for j in range(0,len(bounds_scales[i+1])):
                rad = bounds_scales[i+1][j]*h/np.pi
                arc = patches.Arc((h//2,w//2),rad,rad,0,bounds_angles[i]*180/np.pi,bounds_angles[i+1]*180/np.pi,color = 'r',Fill = 0)
                ax.add_patch(arc)
                arc2 = patches.Arc((h//2,w//2),rad,rad,0,180+bounds_angles[i]*180/np.pi,180+bounds_angles[i+1]*180/np.pi,color = 'r',Fill = 0)
                ax.add_patch(arc2)
        #Plot arcs for last angular sector
        for i in range(0,len(bounds_scales[-1])):
            rad = bounds_scales[-1][i]*h/np.pi
            arc = patches.Arc((h//2,w//2),rad,rad,0,bounds_angles[-1]*180/np.pi,180+bounds_angles[1]*180/np.pi,color = 'r',Fill = 0)
            ax.add_patch(arc)
            arc2 = arc = patches.Arc((h//2,w//2),rad,rad,0,180+bounds_angles[-1]*180/np.pi,360+bounds_angles[1]*180/np.pi,color = 'r',Fill = 0)
            ax.add_patch(arc2)
    else:
        return -1;
    plt.show()
    