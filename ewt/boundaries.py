import numpy as np
from scipy.special import iv, erf, erfinv
from ewt.utilities import *


"""
ewt_boundariesDetect(absf, params)
Adaptively detects boundaries in 1D magnitude Fourier spectrum based on the 
detection method chosen in params.detect
Input:
    absf    - magnitude Fourier spectrum
    params  - parameters for EWT (see utilities)
Output:
    bounds     - resulting boundaries in index domain
Author: Basile Hurat, Jerome Gilles"""
def ewt_boundariesDetect(absf, params, sym = 1):
    if params.log == 1:     #apply log parameter
        preproc = np.log(absf)
    else:
        preproc = np.copy(absf)
    if params.removeTrends.lower() != 'none':   #apply removeTrend parameter
        preproc = removeTrends(absf, params)
    if params.spectrumRegularize.lower() != 'none': #apply spectrumRegularize parameter
        preproc = spectrumRegularize(preproc, params)
    
    #Choose detection method
    if params.detect == 'scalespace':
        bounds = ewt_GSSDetect(preproc, params, sym)
    elif params.detect == 'locmax':
        if sym == 1:
            bounds = ewt_localMaxBounds(preproc[0:len(preproc) // 2], params.N)
        else:
            bounds = ewt_localMaxBounds(preproc, params.N)
    elif params.detect == 'locmaxmin':
        if sym == 1:
            bounds = ewt_localMaxMinBounds(preproc[0:len(preproc) // 2], params.N)
        else:
            bounds = ewt_localMaxMinBounds(preproc, params.N)
    elif params.detect == 'locmaxminf':
        if sym == 1:
            bounds = ewt_localMaxMinBounds(preproc[0:len(preproc) // 2], params.N,absf[0:len(absf) // 2])
        else:
            bounds = ewt_localMaxMinBounds(preproc, params.N, absf)
    elif params.detect == 'adaptivereg':
        if sym == 1:
            bounds = ewt_adaptiveBounds(preproc[0:len(preproc) // 2], params.init_bounds)
        else:
            bounds = ewt_adaptiveBounds(preproc, params.init_bounds)        
    elif params.detect == 'adaptive':
        if sym == 1:
            bounds = ewt_adaptiveBounds(preproc[0:len(preproc) // 2], params.init_bounds,absf[0:len(absf) // 2])
        else:
            bounds = ewt_adaptiveBounds(preproc, params.init_bounds, absf)
    for i in range(0, len(bounds)):
        if bounds[i] == 0:
            bounds = np.delete(bounds, i)
            break
    return bounds

"""
ewt_localMaxBounds(f, N)
Detects N highest maxima, and returns the midpoints between them as detected 
boundaries
Input:
    f       - signal to detect maxima from (generally pre-processed magnitude spectrum)
    N       - number of maxima to detect
Output:
    bounds  - resulting detected bounds in index domain
Author: Basile Hurat, Jerome Gilles"""
def ewt_localMaxBounds(f, N):
    #Detect maxima
    maxima = localmin(-f).astype(bool)
    index = np.arange(0, len(maxima))
    maxindex = index[maxima]
    #If we have more than N, keep only N highest maxima values
    if N < len(maxindex):
        order = np.argsort(f[maxima])[-N:]  
        maxindex = np.sort(maxindex[order])
    else:
        N = len(maxindex) - 1
    #find midpoints
    bounds = np.zeros(N)
    bounds[0] = round(maxindex[0] / 2)
    for i in range(0, N-1):
        bounds[i + 1] = (maxindex[i] + maxindex[i + 1]) // 2
    return bounds

"""
ewt_localMaxMinBounds(f, N,f_orig)
Detects N highest maxima, and returns the lowest minima between them as detected 
boundaries
Input:
    f       - signal to detect maxima and minima from (generally pre-processed 
            magnitude spectrum)
    N       - number of maxima to detect
    f_orig  - (Optional) If given, detects minima from this instead of f
Output:
    bounds  - resulting detected bounds in index domain
Author: Basile Hurat, Jerome Gilles"""
def ewt_localMaxMinBounds(f, N, f_orig = []): 
    #Get both maxima and minima of signal
    maxima = localmin(-f).astype(bool)
    if len(f_orig) == 0:
        minima = localmin(f).astype(bool)
    else:
        minima = localmin(f_orig).astype(bool)
    index = np.arange(0, len(maxima))
    maxindex = index[maxima]
    minindex = index[minima]
    
    #If we have more than N, keep only N highest maxima values
    if N<len(maxindex):
        order = np.argsort(f[maxima])[-N:]  
        maxindex = np.sort(maxindex[order])
    else:
        N = len(maxindex) - 1
    
    bounds = np.zeros(N)
    intervalmin = minindex[minindex < maxindex[0]]
    if not len(intervalmin) == 0:
        bounds[0] = intervalmin[np.argmin(f[intervalmin])]
    
    for i in range(0,N - 1):
        intervalmin = minindex[minindex > maxindex[i]]
        intervalmin = intervalmin[intervalmin < maxindex[i + 1]]
        bounds[i + 1] = intervalmin[np.argmin(f[intervalmin])]
    return bounds

"""
ewt_adaptiveBounds(f, N,f_orig)
Adaptively detect from set of initial bounds. Returns lowest minima within a 
neighborhood of given bounds
Input:
    f               - signal to detect maxima and minima from (generally 
                    pre-processed magnitude spectrum)
    init_bounds0    - initial bounds to look at  detection
    f_orig          - (Optional) If given, detects minima from this instead of f
Output:
    bounds          - resulting detected bounds in index domain
Author: Basile Hurat, Jerome Gilles"""
def ewt_adaptiveBounds(f, init_bounds0, f_orig=[]):
    if len(f_orig) != 0:
        f = np.copy(f_orig)
    init_bounds = []
    init_bounds[:] = init_bounds0
    init_bounds.insert(0, 0)
    init_bounds.append(len(f))
    bounds = np.zeros(len(init_bounds) - 1)
    for i in range(0,len(init_bounds) - 1):
        neighb_low = round(init_bounds[i + 1] - round(abs(init_bounds[i + 1] - init_bounds[i])) / 2)
        neighb_high = round(init_bounds[i + 1] + round(abs(init_bounds[i + 1] - init_bounds[i])) / 2)
        bounds[i] = np.argmin(f[neighb_low:neighb_high + 1])
    return np.unique(bounds)

"""
ewt_GSSDetect(f, params,sym)
Detects boundaries using scale-space. 
Input:
    f       - signal to detect boundaries between
    params  - parameters for EWT (see utilities). Notably, the adaptive 
            threshold from params.typeDetect
    sym     - parameter whether or not the signal is symmetric. If true, 
            returns bounds less than middle index
Output:
    bounds  - resulting detected bounds in index domain
Author: Basile Hurat, Jerome Gilles"""  
def ewt_GSSDetect(f, params, sym):
    #Apply gaussian scale-space
    plane = GSS(f)
    #Get persistence (lengths) and indices of minima
    [lengths, indices] = lengthScaleCurve(plane)
    if sym == 1:
        lengths = lengths[indices < len(f) / 2 - 1] #Halve the spectrum
        indices= indices[indices < len(f) / 2 - 1] #Halve the spectrum    
    #apply chosen thresholding method
    if params.typeDetect.lower() == 'otsu':    
        thresh = otsu(lengths)
        bounds = indices[lengths >= thresh]

    elif params.typeDetect.lower() == 'mean':   
        thresh = np.ceil(np.mean(lengths))
        bounds = indices[lengths >= thresh]

    elif params.typeDetect.lower() == 'empiricallaw':
        thresh = empiricalLaw(lengths)
        bounds = indices[lengths >= thresh]

    elif params.typeDetect.lower() == 'halfnormal':
        thresh = halfNormal(lengths)
        bounds = indices[lengths >= thresh]

    elif params.typeDetect.lower() == 'kmeans':
        clusters = ewtkmeans(lengths, 1000)
        upper_cluster = clusters[lengths == max(lengths)][0]
        bounds = indices[clusters == upper_cluster]        

    return bounds


"""
GSS(f)
performs discrete 1D scale-space of signal and tracks minima through 
scale-space
Input:
    f       - input signal
Output:
    plane   - 2D plot of minima paths through scale-space representation of f
Author: Basile Hurat, Jerome Gilles"""
def GSS(f):
    t = 0.5
    n = 3
    num_iter = 1 * np.max([np.ceil(len(f) / n), 3])
    #First, define scale-space kernel (discrete Gaussian kernel)
    ker = np.exp(-t) * iv(np.arange(-n, n + 1), t)
    ker = ker/np.sum(ker)

    #Initialize place to store result of each layer GSS
    plane = np.zeros([len(f), num_iter.astype(int) + 1])
    plane[:, 0] = localmin(f)

    #Iterate through scalespace and store minima at each scale
    for i in range(1, num_iter.astype(int) + 1):
        f = np.pad(f, n, 'reflect')
        f = np.convolve(f, ker, 'same')
        f = f[n:-n]
        plane[:, i] = localmin(f)
        if np.sum(plane[:, i]) <= 2:
            break
    return plane

"""
lengthScaleCurve(plane)
Given the 2D plot of minima paths in scale-space representation, this function
extracts the persistence of each minima, as well as their starting position in 
signal
Input:
    plane   - 2D plot of minima paths through scale-space representation
Output:
    lengths - persistence of each minimum
    indices - position of each minimum
Author: Basile Hurat, Jerome Gilles"""
def lengthScaleCurve(plane):
    [w,num_iter] = plane.shape
    num_curves = np.sum(plane[:, 0])
    lengths = np.ones(num_curves.astype(int))
    indices = np.zeros(num_curves.astype(int))
    current_curve = 0
 
    for i in range(0, w):
        if plane[i, 0] == 1:
            indices[current_curve] = i
            i0 = i
            height = 2
            stop = 0
            while stop == 0:
                flag = 0
                for p in range(-1, 2):
                    if (i + p  < 0) or (i + p >= w):
                        continue
                    #If minimum at next iteration of scale-space, increment length
                    #height, minimum location
                    if plane[i + p, height] == 1:                         
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


"""
localmin(f):
Givan an array f, returns a boolean array that represents positions of local 
minima - Note, handles minima plateaus by assigning all points on plateau as 
minima
Input:
    f       - an array of numbers
Output:
    minima  - boolean array of same length as f, which represents positions of 
            local minima in f
Author: Basile Hurat, Jerome Gilles"""
def localmin(f):
    w = len(f)
    minima = np.zeros(w)
    for i in range(0, w):
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
                    if f[i + left] < f[i]:
                        minima[i] = 0
                        break
                    elif f[i + left] == f[i]:
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
            while (j + 1 < w) and (minima[j + 1] == 1):
                minima[j] = 0
                flat_count += 1
                j += 1
                flag = 1
                
            if flag == 1:
                minima[j - np.floor(flat_count/2).astype(int)] = 1
                minima[j] = 0
                i = j
        i += 1
    minima = removePlateaus(minima)
    minima[0] = 0
    minima[-1] = 0
    return minima

def removePlateaus(x):
    w = len(x); i = 0
    flag = 0
    while i < w:
        if x[i] == 1:
            plateau = 1
            while 1:
                if i + plateau < w and x[i + plateau] == 1:
                    plateau += 1
                    print(f'{i}, plateau = {plateau}')
                else:
                    flag = plateau > 1
                    break
        if flag:
            x[i:i + plateau] = 0
            x[i + plateau // 2] = 1
            i = i + plateau
        else:
            i += 1
    return x
    
"""
otsu(lengths)
2-class classification method which minimizes the inter-class variance of the 
class probability and the respective class averages
Input:
    lengths - array to be thresholded or classified
Output:
    thresh  - detected threshold that separates the two classes
Author: Basile Hurat, Jerome Gilles"""
def otsu(lengths):
    hist_max = np.max(lengths); 
    histogram = np.histogram(lengths, hist_max.astype(int))[0]
    hist_normalized = histogram / np.sum(histogram) #normalize
    Q = hist_normalized.cumsum()
    
    bins = np.arange(hist_max)
    fn_min = np.inf
    thresh = -1

    for i in range(1, hist_max.astype(int)):
        p1, p2 = np.hsplit(hist_normalized, [i])
        q1, q2 = Q[i], Q[hist_max.astype(int) - 1] - Q[i]
        if q1 < 1.e-6 or q2 < 1.e-6:
            continue
        b1, b2 = np.hsplit(bins, [i]) #weights

        #Means and variances
        m1 = np.sum(p1 * b1) / q1
        m2 = np.sum(p2 * b2) / q2
        v1 = np.sum((b1 - m1) ** 2 * p1) / q1
        v2 = np.sum((b2 - m2) ** 2 * p2) / q2

        #calculates the minimization function
        fn = v1 * q1 + v2 * q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    return thresh

"""
empiricalLaw(lengths)
2-class classification method which classifies by considering lengths which are
epsilon meaningful for an empirical law. 
Input:
    lengths     - array to be thresholded or classified
Output:
    meaningful  - boolean array of where meaningful lengths are
Author: Basile Hurat, Jerome Gilles"""
def empiricalLaw(lengths):
    hist_max = np.max(lengths); 
    histogram = np.histogram(lengths, hist_max.astype(int))[0]
    hist_normalized = histogram / np.sum(histogram) #normalize
    Q = hist_normalized.cumsum()
    meaningful = np.where(Q > (1 - 1 / len(lengths)))[0][0] + 1
    return meaningful
    
"""
halfNormal(lengths)
2-class classification method which classifies by considering lengths which are
epsilon-meaningful for a half-normal law fitted to the data. 
Input:
    lengths     - array to be thresholded or classified
Output:
    thresh  - detected threshold that separates the two classes
Author: Basile Hurat, Jerome Gilles"""
def halfNormal(lengths):
    sigma=np.sqrt(np.pi) * np.mean(lengths)
    thresh = sigma * erfinv(erf(np.max(lengths) / sigma) - 1 / len(lengths))
    return thresh

"""
ewtkmeans(lengths)
1D k-means clustering function for 2 class classification
Input:
    lengths     - array to be clustered or classified
Output:
    closest  - label array that gives final classification 
Author: Basile Hurat, Jerome Gilles"""
def ewtkmeans(lengths,maxIter):
    k = 2
    centroids = np.zeros(k)
    distances = np.inf * np.ones([len(lengths), k])
    closest = -np.ones([len(lengths), 2])
    closest[:, 0] = distances[:, 0]
    breaker = 0
    for i in range(0, k):
        centroids[i] = np.random.uniform(1, np.max(lengths))
    while breaker < maxIter:
        prev_closest = closest[:, 1]
        for i in range(0, k):
            distances[:, i] = np.abs(lengths - centroids[i])
            closest[distances[:, i] < closest[:,0], 1] = i
            closest[distances[:, i] < closest[:,0], 0] = distances[distances[:, i] < closest[:, 0], i]
        if np.all(closest[:, 1] == prev_closest):
            break
        for i in range(0, i):
            centroids[i] = np.mean(lengths[closest[:, 1] == i])
        if breaker == maxIter - 1:
            print('k-means did not converge')
        breaker += 1
    return closest[:, 1]
        