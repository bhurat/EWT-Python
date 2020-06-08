def EWT_LP_Wavelet(wn,wm,aw,gamma,N):
    if wn > np.pi:  #If greater than pi, subtract 2pi, otherwise dont
        a = 1;
    else:
        a = 0

    mbn = wn - gamma*abs(wn - a*2*np.pi) #beginning of first transition
    pbn = wn + gamma*abs(wn - a*2*np.pi) #end of first transition
    an = 1/(2*gamma*abs(wn - a*2*np.pi)) #scaling in first transition's beta function

    if wm > np.pi: #If greater than pi, subtract 2pi, otherwise dont
        a=1;
    else:
        a=0;
    
    mbm = wm - gamma*abs(wm - a*2*np.pi); #beginning of second transition
    pbm = wm + gamma*abs(wm - a*2*np.pi); #end of second transition
    am = 1/(2*gamma*abs(wm - a*2*np.pi));  #scaling in second transition's beta function
    
    ymw = 1.0*(aw >= pbn)*(aw<= mbm) #equals 1 between transition areas
    ymw += (aw > mbn)*(aw < pbn)*np.sin(np.pi*EWT_beta(an*(aw - mbn))/2) #1st transition area
    if wm < np.pi:
        ymw += (aw > mbm)*(aw < pbm)*np.cos(np.pi*EWT_beta(am*(aw - mbm))/2) #2nd transition area
    else:
        ymw += (aw > mbm)*(aw < pbm)*1.0
    return ymw
