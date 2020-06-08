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

