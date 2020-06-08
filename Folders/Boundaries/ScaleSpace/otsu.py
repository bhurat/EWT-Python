def Otsu(lengths):
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