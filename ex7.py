# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 19:32:07 2020

@author: gio-x
"""

import numpy.random as rnd
import numpy as np
from numpy.fft import rfftfreq, irfft
import pandas as pd
import statsfuncs
import mfdfa
import matplotlib.pyplot as plt

def pmodel(seriestype):
    if(seriestype=="Endogenous"):
        p=0.32 + 0.1*rnd.uniform()
        slope=0.4
    else:
        p=0.18 + 0.1*rnd.uniform()
        slope=0.7
    noValues=8192
    noOrders = int(np.ceil(np.log2(noValues)))
    
    y = np.array([1])
    for n in range(noOrders):
        y = next_step_1d(y, p)
    
    if (slope):
        fourierCoeff = fractal_spectrum_1d(noValues, slope/2)
        meanVal = np.mean(y)
        stdy = np.std(y)
        x = np.fft.ifft(y - meanVal)
        phase = np.angle(x)
        x = fourierCoeff*np.exp(1j*phase)
        x = np.fft.fft(x).real
        x *= stdy/np.std(x)
        x += meanVal
    else:
        x = y
    return y[0:noValues]


def next_step_1d(y, p):
    y2 = np.zeros(y.size*2)
    sign = np.random.rand(1, y.size) - 0.5
    sign /= np.abs(sign)
    y2[0:2*y.size:2] = y + sign*(1-2*p)*y
    y2[1:2*y.size+1:2] = y - sign*(1-2*p)*y
    
    return y2


def fractal_spectrum_1d(noValues, slope):
    ori_vector_size = noValues
    ori_half_size = ori_vector_size//2
    a = np.zeros(ori_vector_size)
    
    for t2 in range(ori_half_size):
        index = t2
        t4 = 1 + ori_vector_size - t2
        if (t4 >= ori_vector_size):
            t4 = t2
        coeff = (index + 1)**slope
        a[t2] = coeff
        a[t4] = coeff
        
    a[1] = 0
    
    return a

def powerlaw_psd_gaussian(exponent, size=8192, fmin=0):
    """Gaussian (1/f)**beta noise.
    Based on the algorithm in:
    Timmer, J. and Koenig, M.:
    On generating power law noise.
    Astron. Astrophys. 300, 707-710 (1995)
    Normalised to unit variance
    Parameters:
    -----------
    exponent : float
        The power-spectrum of the generated noise is proportional to
        S(f) = (1 / f)**beta
        flicker / pink noise:   exponent beta = 1
        brown noise:            exponent beta = 2
        Furthermore, the autocorrelation decays proportional to lag**-gamma
        with gamma = 1 - beta for 0 < beta < 1.
        There may be finite-size issues for beta close to one.
    shape : int or iterable
        The output has the given shape, and the desired power spectrum in
        the last coordinate. That is, the last dimension is taken as time,
        and all other components are independent.
    fmin : float, optional
        Low-frequency cutoff.
        Default: 0 corresponds to original paper. It is not actually
        zero, but 1/samples.
    Returns
    -------
    out : array
        The samples.
    Examples:
    ---------
    # generate 1/f noise == pink noise == flicker noise
    >>> import colorednoise as cn
    >>> y = cn.powerlaw_psd_gaussian(1, 5)
    """
    
    # Make sure size is a list so we can iterate it and assign to it.
    try:
        size = list(size)
    except TypeError:
        size = [size]
    
    # The number of samples in each time series
    samples = size[-1]
    
    # Calculate Frequencies (we asume a sample rate of one)
    # Use fft functions for real output (-> hermitian spectrum)
    f = rfftfreq(samples)
    
    # Build scaling factors for all frequencies
    s_scale = f
    fmin = max(fmin, 1./samples) # Low frequency cutoff
    ix   = np.sum(s_scale < fmin)   # Index of the cutoff
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale**(-exponent/2.)
    
    # Calculate theoretical output standard deviation from scaling
    w      = s_scale[1:].copy()
    w[-1] *= (1 + (samples % 2)) / 2. # correct f = +-0.5
    sigma = 2 * np.sqrt(np.sum(w**2)) / samples
    
    # Adjust size to generate one Fourier component per frequency
    size[-1] = len(f)

    # Add empty dimension(s) to broadcast s_scale along last
    # dimension of generated random power + phase (below)
    dims_to_add = len(size) - 1
    s_scale     = s_scale[(np.newaxis,) * dims_to_add + (Ellipsis,)]
    
    # Generate scaled random power + phase
    sr = rnd.normal(scale=s_scale, size=size)
    si = rnd.normal(scale=s_scale, size=size)
    
    # If the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    if not (samples % 2): si[...,-1] = 0
    
    # Regardless of signal length, the DC component must be real
    si[...,0] = 0
    
    # Combine power + corrected phase to Fourier components
    s  = sr + 1J * si
    
    # Transform to real time series & scale to unit variance
    y = irfft(s, n=samples, axis=-1) / sigma
    x=range(0,len(y))
    return y

def randomseries(n):
    '''
Gerador de Série Temporal Estocástica - V.1.2 por R.R.Rosa 
Trata-se de um gerador randômico não-gaussiano sem classe de universalidade via PDF.
Input: n=número de pontos da série
res: resolução 
    '''
    res = n/12
    df = pd.DataFrame(np.random.randn(n) * np.sqrt(res) * np.sqrt(1 / 128.)).cumsum()
    a=df[0].tolist()
    a=statsfuncs.normalize(a)
    x=range(0,n)
    return a

def Logistic(dummy="nada"):
    N=8192
    rho=3.85 + 0.15*np.random.uniform()
    tau = 1.1
    x = [0.001]
    y = [0.001]
    for i in range(1,N):
      y.append( tau*x[-1] )
      x.append( rho*x[-1]*(1.0-x[-1]))
    return x

def HenonMap(dummy="nada"):
    N=8192
    a=1.350 + 0.05*np.random.uniform()
    b=0.21 + 0.08*np.random.uniform()
    x = [0.1]
    y = [0.3]
    for i in range(1,N):
        y.append(b * x[-1])
        x.append(y[-2] + 1.0 - a *x[-1]*x[-1])
    return y


grng=[]
endo=[]
exo=[]
logis=[]
henon=[]
white=[]
pink=[]
red=[]

plt.title("Espectro de singularidade: GRNG")
for i in range(30):
    a,b,c=mfdfa.makemfdfa(randomseries(8192), False)
    plt.plot(b,c, 'ko-')
plt.xlabel(r"$\alpha$")
plt.legend()
plt.ylabel(r"f($\alpha$)")
plt.show()
plt.title("Espectro de singularidade: pmodel")
for i in range(15):
    if  i==0:
        a,b,c=mfdfa.makemfdfa(pmodel("Endogenous"))
        plt.plot(b,c, 'ro-', label="Endogenous")
        a,b,c=mfdfa.makemfdfa(pmodel("Exogenous"))
        plt.plot(b,c, 'bo-', label="Exogenous")
    else:
        a,b,c=mfdfa.makemfdfa(pmodel("Endogenous"))
        plt.plot(b,c, 'ro-')
        a,b,c=mfdfa.makemfdfa(pmodel("Exogenous"))
        plt.plot(b,c, 'bo-')
plt.xlabel(r"$\alpha$")
plt.legend()
plt.ylabel(r"f($\alpha$)")
plt.show()

plt.title("Espectro de singularidade: Chaos Noise")
for i in range(15):
    if i == 0:
        a,b,c=mfdfa.makemfdfa(Logistic())
        plt.plot(b,c, 'ko-', label="Logistic")
        a,b,c=mfdfa.makemfdfa(HenonMap())
        plt.plot(b,c, 'bo-', label="Henon")
    else:
        a,b,c=mfdfa.makemfdfa(Logistic())
        plt.plot(b,c, 'ko-')
        a,b,c=mfdfa.makemfdfa(HenonMap())
        plt.plot(b,c, 'bo-')
plt.xlabel(r"$\alpha$")
plt.legend()
plt.ylabel(r"f($\alpha$)")
plt.show()

plt.title("Espectro de singularidade: Colored Noise")
for i in range(10):
    if i == 0:
        a,b,c=mfdfa.makemfdfa(powerlaw_psd_gaussian(0) )
        plt.plot(b,c, 'ko-',label="white")
        a,b,c=mfdfa.makemfdfa(powerlaw_psd_gaussian(1))
        plt.plot(b,c, 'o-', c="pink", label="pink")
        a,b,c=mfdfa.makemfdfa(powerlaw_psd_gaussian(2))
        plt.plot(b,c, 'ro-', label="red")
    else:
        a,b,c=mfdfa.makemfdfa(powerlaw_psd_gaussian(0))
        plt.plot(b,c, 'ko-')
        a,b,c=mfdfa.makemfdfa(powerlaw_psd_gaussian(1))
        plt.plot(b,c, 'o-', c="pink")
        a,b,c=mfdfa.makemfdfa(powerlaw_psd_gaussian(2))
        plt.plot(b,c, 'ro-')
plt.xlabel(r"$\alpha$")
plt.legend()
plt.ylabel(r"f($\alpha$)")
plt.show()
