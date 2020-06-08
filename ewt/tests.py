import numpy as np

def EWT_1DEx_sig2(N):
    dt = 1/N
    t = np.arange(0,1+dt,dt)
    sig = 6*t**2 + np.cos(10*np.pi*t + 10*np.pi*t**2)
    sig[t > 0.5] += np.cos(80*np.pi*t[t > 0.5] - 15*np.pi)
    sig[t <= 0.5] += np.cos(60*np.pi*t[t <= 0.5])
    return sig