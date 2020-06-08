def EWT_LP_Scaling(w1,aw,gamma,N):
    mbn = (1 - gamma)*w1 #beginning of transitino
    pbn = (1 + gamma)*w1 #end of transition
    an = 1/(2*gamma*w1) #scaling in beta function

    yms = 1.0*(aw <= mbn) #if less than lower bound, equals 1
    yms += (aw > mbn)*(aw <= pbn)*np.cos(np.pi*EWT_beta(an*(aw - mbn))/2) #Transition area
    return yms