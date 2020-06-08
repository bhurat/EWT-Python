def EWT_Filterbank(bounds,N):
    #Calculate Gamma
    gamma = 1;
    for i in range(0,len(bounds)-1):
        r = (bounds[i+1] - bounds[i])/(bounds[i+1]+bounds[i])
        if r < gamma:
            gamma = r
    aw = np.arange(0,2*np.pi,2*np.pi/N)
    aw[np.floor(N/2).astype(int):] -= 2*np.pi 
    aw = np.abs(aw)
    filterbank = np.zeros([N, len(bounds)+1])
    filterbank[:,0] = EWT_LP_Scaling(bounds[0],aw,gamma,N)
    for i in range(1,len(bounds)):
        filterbank[:,i] = EWT_LP_Wavelet(bounds[i-1],bounds[i], aw, gamma, N)
    filterbank[:,len(bounds)] = EWT_LP_Wavelet(bounds[len(bounds)-1],np.pi,aw,gamma,N)
    return filterbank