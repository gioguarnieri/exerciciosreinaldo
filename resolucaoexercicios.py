# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 10:10:23 2020

@author: gio-x
"""
'''
###############################################################################
IMPORTAÇÃO DE MÓDULOS
##############################################################################
'''


# import matplotlib.mlab as mlab
from scipy.stats import norm
# from scipy import optimize
import numpy.random as rnd
from numpy.fft import rfftfreq, irfft
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# from pylab import savefig
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon
from yellowbrick.cluster import KElbowVisualizer
import mfdfa
import statsfuncs

'''
Função que serviu para teste do mapa de cullen frey, gera números aleatórios
numa certa distribuição e o mapa posiciona naquela distribuição de acordo
com os momentos estatísticos de Kurtosis e Skewness²
def teste(N):
    x=range(N)
    y=[]
    for i in x:
        y.append(rnd.lognormal())
    return x,y
'''
'''
###############################################################################

FUNÇÕES QUE GERAM MODELOS

Seção do código de funções enviadas pelo professor, modificadas para caber 
em um código que faça tudo de uma vez

###############################################################################
'''
def pmodel(seriestype):
    if(seriestype=="Endogenous"):
        p=0.32 + 0.1*rnd.uniform()
    else:
        p=0.18 + 0.1*rnd.uniform()
    noValues=8192
    slope=0.4
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
    return x[0:noValues], y[0:noValues]


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
    return x,y

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
    return x,a

def Logistic(dummy):
    N=8192
    rho=3.85 + 0.15*np.random.uniform()
    tau = 1.1
    x = [0.001]
    y = [0.001]
    for i in range(1,N):
      y.append( tau*x[-1] )
      x.append( rho*x[-1]*(1.0-x[-1]))
    return y,x

def HenonMap(dummy):
    N=8192
    a=1.350 + 0.05*np.random.uniform()
    b=0.21 + 0.08*np.random.uniform()
    x = [0.1]
    y = [0.3]
    for i in range(1,N):
        y.append(b * x[-1])
        x.append(y[-2] + 1.0 - a *x[-1]*x[-1])
    return x,y





'''
###############################################################################

FUNÇÕES CONSTRUTORAS DE SÉRIES OU ESPAÇOS

Funções que servem para construir as séries temporais ou construir os k-means
geralmente retornam os dados que elas geram.

###############################################################################
'''

def makeseries(func, iterationlist, amount):
    values=[]
    ilist=[]
    rawdata=[]
    for i in iterationlist:
        for j in range(amount):
            x,y=func(i)
            alfa,xdfa,ydfa, reta = statsfuncs.dfa1d(y,1)
            freqs, power, xdata, ydata, amp, index, powerlaw, INICIO, FIM = statsfuncs.psd(y)
            psi=mfdfa.makemfdfa(y)
            values.append([statsfuncs.variance(y), statsfuncs.skewness(y), statsfuncs.kurtosis(y)+3, alfa, index,psi])
            ilist.append(i)
        rawdata.append([i,x,y, alfa, xdfa, ydfa, reta, freqs, power, xdata, ydata, amp, index, powerlaw, INICIO, FIM])
    return values, ilist, rawdata

def makeK(d,ilist, title):
    d=np.array(d)
    kk=pd.DataFrame({'Variance': d[:,0], 'Skewness': d[:,1], 'Kurtosis': d[:,2]})
    K=20
    model=KMeans()
    visualizer = KElbowVisualizer(model, k=(1,K))
    kIdx=visualizer.fit(kk)        # Fit the data to the visualizer
    visualizer.show()        # Finalize and render the figure
    kIdx=kIdx.elbow_value_
    model=KMeans(n_clusters=kIdx).fit(kk)
    # scatter plot
    fig = plt.figure()
    ax = Axes3D(fig) #.add_subplot(111))
    cmap = plt.get_cmap('gnuplot')
    clr = [cmap(i) for i in np.linspace(0, 1, kIdx)]
    for i in range(0,kIdx):
        ind = (model.labels_==i)
        ax.scatter(d[ind,2],d[ind,1], d[ind,0], s=30, c=clr[i], label='Cluster %d'%i)
    
    ax.set_xlabel("Kurtosis")
    ax.set_ylabel("Skew")
    ax.set_zlabel("Variance")
    plt.title(title+': KMeans clustering with K=%d' % kIdx)
    plt.legend()
    plt.savefig(title+"clustersnoises.png")
    plt.show()
    d=pd.DataFrame({'Variance': d[:,0], 'Skewness': d[:,1], 'Kurtosis': d[:,2], 'Alpha': d[:,3], 'Beta': d[:,4], "Psi": d[:,5], "Cluster": model.labels_}, index=ilist)
    return d




'''
###############################################################################

FUNÇÕES QUE GERAM GRÁFICOS

Funções que não retornam nada e só fazem gráficos

###############################################################################
'''

def cullenfrey(xd,yd,legend, title):
    plt.figure(num=None, figsize=(8, 8), dpi=100, facecolor='w', edgecolor='k')
    fig, ax = plt.subplots()
    maior=max(xd)
    polyX1=maior if maior > 4.4 else 4.4
    polyY1=polyX1+1
    polyY2=3/2.*polyX1+3
    y_lim = polyY2 if polyY2 > 10 else 10
    
    x = [0,polyX1,polyX1,0]
    y = [1,polyY1,polyY2,3]
    scale = 1
    poly = Polygon( np.c_[x,y]*scale, facecolor='#1B9AAA', edgecolor='#1B9AAA', alpha=0.5)
    ax.add_patch(poly)
    ax.plot(xd,yd, marker="o", c="#e86a92", label=legend, linestyle='')
    ax.plot(0, 4.187999875999753, label="logistic", marker='+', c='black')
    ax.plot(0, 1.7962675925351856, label ="uniform", marker='^',c='black')
    ax.plot(4, 9, label="exponential", marker='s', c='black')
    ax.plot(0, 3, label="normal", marker='*',c='black')
    ax.plot(np.arange(0,polyX1,0.1), 3/2.*np.arange(0,polyX1,0.1)+3, label="gamma", linestyle='-',c='black')
    ax.plot(np.arange(0,polyX1,0.1), 2*np.arange(0,polyX1,0.1)+3, label="lognormal", linestyle='-.',c='black')
    ax.legend()
    ax.set_ylim(y_lim,0)
    ax.set_xlim(-0.2,polyX1)
    plt.xlabel("Skewness²")
    plt.title(title+": Cullen and Frey map")
    plt.ylabel("Kurtosis")
    plt.savefig(title+legend+"cullenfrey.png")
    plt.show()

def makespaces(s2, k, alpha, beta, legend, title):
    kk=pd.DataFrame({'Skew²': s2, 'Kurtosis': k, 'Alpha': alpha, 'Beta': beta})
    K=8
    model=KMeans()
    visualizer = KElbowVisualizer(model, k=(1,K))
    kIdx=visualizer.fit(kk.drop(columns="Beta"))        # Fit the data to the visualizer
    visualizer.show()        # Finalize and render the figure
    kIdx=kIdx.elbow_value_
    model=KMeans(n_clusters=kIdx).fit(kk.drop(columns="Beta"))
    fig = plt.figure()
    ax = Axes3D(fig)
    cmap = plt.get_cmap('gnuplot')
    clr = [cmap(i) for i in np.linspace(0, 1, kIdx)]
    for i in range(0,kIdx):
        ind = (model.labels_==i)
        ax.scatter(kk["Skew²"][ind],kk["Kurtosis"][ind], kk["Alpha"][ind], s=30, c=clr[i], label='Cluster %d'%i)
    ax.set_xlabel("Skew²")
    ax.set_ylabel("Kurtosis")
    ax.set_zlabel(r"$\alpha$")
    ax.legend()
    plt.title(title+": EDF-K-means")
    plt.show()
    model=KMeans()
    visualizer = KElbowVisualizer(model, k=(1,K))
    kIdx=visualizer.fit(kk.drop(columns="Alpha"))        # Fit the data to the visualizer
    visualizer.show()        # Finalize and render the figure
    kIdx=kIdx.elbow_value_
    model=KMeans(n_clusters=kIdx).fit(kk.drop(columns="Alpha"))
    fig = plt.figure()
    ax = Axes3D(fig)
    cmap = plt.get_cmap('gnuplot')
    clr = [cmap(i) for i in np.linspace(0, 1, kIdx)]
    for i in range(0,kIdx):
        ind = (model.labels_==i)
        ax.scatter(kk["Skew²"][ind],kk["Kurtosis"][ind], kk["Beta"][ind], s=30, c=clr[i], label='Cluster %d'%i)
    ax.set_xlabel("Skew²")
    ax.set_ylabel("Kurtosis")
    ax.set_zlabel(r"$\beta$")
    ax.legend()
    plt.title(title+": EPSB-K-means")
    plt.show()



'''
###############################################################################

FUNÇÃO PRINCIPAL (main)

Main é onde eu construo a maioria dos gráficos, é feito separadamente em vez
de fazer em uma só função pois exercícios diferentes pedem gráficos diferentes.
Então ajustei cada tipo de acordo com a necessidade.

A função pede um input, com a ideia de fazer tudo que se pede em um click.

###############################################################################
'''
def main():
    choice=input("Qual distribuição escolher:\n1 - GRNG\n2 - Colored Noise\n3 - pmodel\n4 - Chaos Noise \n5 - Ler de um arquivo\n6 - Sair\n\n")
    if(choice=="1"):
        '''
    Começando pela série GRNG, serão gerados 8 famílias de sinais com elementos 2^n
    sendo que n varia de 6 até 13, ou, 64 a 8192 e para cada família serão gerados
    10 sinais diferentes e guardados todos eles numa lista rawdata, seus momentos
    estatísticos serão guardados em d, e ilist é uma lista de iteração para cada N
    usada para identificar corretamente os valores no DataFrame que será gerado.
    
    Feito esse conjunto de dados, serão feitas as análises, começando pelo Detrended
    Fluctuation Analysis e o Power Spectrum Density, que serão mostrados em um gráfico.
    Essas duas análises serão feitas só para um de cada família gerada.
    
    Depois disso será feito o agrupamento dos momentos estatísticos pelo método
    K-Means, que fará um gráfico mostrando o agrupamento, e finalizando fazendo 
    o mapeamento de Cullen e Frey, um para cada família das séries.
        '''
        title="Série: GRNG. Quantidade de Dados N={0}"
        i=[2**i for i in range(6,14)]
        d,ilist,rawdata=makeseries(randomseries, i,10)
        for i in range(len(rawdata)):
            plt.figure(figsize=(20, 12))
            #Plot da série temporal
            ax1 = plt.subplot(211)
            ax1.set_title(title.format(rawdata[i][0]), fontsize=18)
            ax1.plot(rawdata[i][1],rawdata[i][2], color="firebrick", linestyle='-', label="Data")
            #Plot e cálculo do DFA
            ax2 = plt.subplot(223)
            ax2.set_title(r"Detrended Fluctuation Analysis $\alpha$={0:.3}".format(rawdata[i][3]), fontsize=15)
            ax2.plot(rawdata[i][4],rawdata[i][5], marker='o', linestyle='', color="#12355B", label="{0:.3}".format(rawdata[i][3]))
            ax2.plot(rawdata[i][4], rawdata[i][6], color="#9DACB2")
            #Plot e cáculo do PSD
            ax3 = plt.subplot(224)
            ax3.set_title(r"Power Spectrum Density $\beta$={0:.3}".format(rawdata[i][12]), fontsize=15)
            ax3.set_yscale('log')
            ax3.set_xscale('log')
            ax3.plot(rawdata[i][7], rawdata[i][8], '-', color = 'deepskyblue', alpha = 0.7)
            ax3.plot(rawdata[i][9], rawdata[i][10], color = "darkblue", alpha = 0.8)
            ax3.axvline(rawdata[i][7][rawdata[i][14]], color = "darkblue", linestyle = '--')
            ax3.axvline(rawdata[i][7][rawdata[i][15]], color = "darkblue", linestyle = '--')    
            ax3.plot(rawdata[i][9], rawdata[i][13](rawdata[i][9], rawdata[i][11], rawdata[i][12]),color="#D65108", linestyle='-', linewidth = 3, label = '{0:.3}$'.format(rawdata[i][12])) 
            ax2.set_xlabel("log(s)")
            ax2.set_ylabel("log F(s)")
            ax3.set_xlabel("Frequência (Hz)")
            ax3.set_ylabel("Potência")
            ax3.legend()
            plt.savefig("GRNGserietemporalpsddfa{}.png".format(i))
            plt.show()
        title="GRNG"
        d=makeK(d,ilist, title)

    elif(choice=="2"):
        '''
    De maneira semelhante ao GRNG, serão feitas as mesmas coisas para as famílias
    de ruído colorido White Noise, Pink Noise e Red Noise. Gerando 20 sinais para
    cada uma dessas famílias.
    
    Aqui será feito em adição um histograma que será ajustado numa gaussiana com 
    parâmetros mu e sigma.
        '''
        title="Série: Colored Noise. Expoente = {0}"
        d,ilist,rawdata=makeseries(powerlaw_psd_gaussian, range(0,3), 20)
        while(0 in ilist or 1 in ilist or 2 in ilist):
            ilist[ilist.index(0)] = 'white noise'
            ilist[ilist.index(1)] = 'pink noise'
            ilist[ilist.index(2)] = 'red noise'
        for i in range(len(rawdata)):
            #Plot e ajuste do histograma da série temporal
            (mu,sigma)=norm.fit(rawdata[i][2])
            plt.title((title+"\nMu= {1:.3}, Sigma={2:.3}.").format(rawdata[i][0], mu, sigma))
            n, bins, patches = plt.hist(rawdata[i][2], 60, density=1, facecolor='powderblue', alpha=0.75)
            plt.plot(bins,norm.pdf(bins,mu,sigma), c="black", linestyle='--')
            plt.savefig("colorednoise{}PDF.png".format(i))
            plt.show()
            plt.figure(figsize=(20, 12))
            #Plot da série temporal
            ax1 = plt.subplot(211)
            ax1.set_title(title.format(rawdata[i][0]), fontsize=18)
            ax1.plot(rawdata[i][1],rawdata[i][2],color="firebrick", linestyle='-')
            #Plot e cálculo do DFA
            ax2 = plt.subplot(223)
            ax2.set_title(r"Detrended Fluctuation Analysis $\alpha$={0:.3}".format(rawdata[i][3]), fontsize=15)
            ax2.plot(rawdata[i][4],rawdata[i][5], marker='o', linestyle='', color="#12355B")
            ax2.plot(rawdata[i][4], rawdata[i][6], color="#9DACB2")
            #Plot e cálculo do PSD
            ax3 = plt.subplot(224)
            ax3.set_title(r"Power Spectrum Density $\beta$={0:.3}".format(rawdata[i][12]), fontsize=15)
            ax3.set_yscale('log')
            ax3.set_xscale('log')
            ax3.plot(rawdata[i][7], rawdata[i][8], '-', color = 'deepskyblue', alpha = 0.7)
            ax3.plot(rawdata[i][9], rawdata[i][10], color = "darkblue", alpha = 0.8)
            ax3.axvline(rawdata[i][7][rawdata[i][14]], color = "darkblue", linestyle = '--')
            ax3.axvline(rawdata[i][7][rawdata[i][15]], color = "darkblue", linestyle = '--')    
            ax3.plot(rawdata[i][9], rawdata[i][13](rawdata[i][9], rawdata[i][11], rawdata[i][12]),color="#D65108", linestyle='-', linewidth = 3, label = '$%.4f$' %(rawdata[i][12]))
            ax2.set_xlabel("log(s)")
            ax2.set_ylabel("log F(s)")
            ax3.set_xlabel("Frequência (Hz)")
            ax3.set_ylabel("Potência")
            ax2.set_xlabel("log(s)")
            ax2.set_ylabel("log F(s)")
            ax3.set_xlabel("Frequência (Hz)")
            ax3.set_ylabel("Potência")
            plt.savefig("CNserietemporalpsddfa{}.png".format(i))
            plt.show()
        title="colorednoise"
        d=makeK(d,ilist, title)
        
    elif(choice=="3"):
        '''
    Para o pmodel serão feitas as mesmas análises que o GRNG, com as famílias 
    Endógeno e Exógeno, que geram valores aleatórios para cada família. 
    Valores entre 0.32~0.42 e 0.18~0.28, respectivamente, gerando 30 valores 
    aleatórios onde cada um gera seu sinal totalizando em 60 sinais.
        '''
        qtd=1
        p=[]
        for i in range(qtd):
            p.append("Endogenous")
        for i in range(qtd):
            p.append("Exogenous")
        title="Série: pmodel. {0}"
        d,ilist,rawdata=makeseries(pmodel, p,30)
        for i in range(len(rawdata)):
            plt.figure(figsize=(20, 12))
            #Plot da série temporal
            ax1 = plt.subplot(211)
            ax1.set_title(title.format(rawdata[i][0]), fontsize=18)
            ax1.plot(rawdata[i][2],color="firebrick", linestyle='-')
            #Plot e cálculo do DFA
            ax2 = plt.subplot(223)
            ax2.set_title(r"Detrended Fluctuation Analysis $\alpha$={0:.3}".format(rawdata[i][3]), fontsize=15)
            plt.plot(rawdata[i][4],rawdata[i][5], marker='o', linestyle='', color="#12355B")
            plt.plot(rawdata[i][4], rawdata[i][6], color="#9DACB2")
            #Plot e cálculo do PSD
            ax3 = plt.subplot(224)
            ax3.set_title(r"Power Spectrum Density $\beta=${0:.3}".format(rawdata[i][12]), fontsize=15)
            ax3.set_yscale('log')
            ax3.set_xscale('log')
            ax3.plot(rawdata[i][7], rawdata[i][8], '-', color = 'deepskyblue', alpha = 0.7)
            ax3.plot(rawdata[i][9], rawdata[i][10], color = "darkblue", alpha = 0.8)
            ax3.axvline(rawdata[i][7][rawdata[i][14]], color = "darkblue", linestyle = '--')
            ax3.axvline(rawdata[i][7][rawdata[i][15]], color = "darkblue", linestyle = '--')    
            ax3.plot(rawdata[i][9], rawdata[i][13](rawdata[i][9], rawdata[i][11], rawdata[i][12]), color="#D65108", linestyle='-', linewidth = 3, label = '$%.4f$' %(rawdata[i][12]))
            ax2.set_xlabel("log(s)")
            ax2.set_ylabel("log F(s)")
            ax3.set_xlabel("Frequência (Hz)")
            ax3.set_ylabel("Potência")
            plt.savefig("Pmodelserietemporalpsddfa{}.png".format(i))
            plt.show()
        title="pmodel"
        d=makeK(d,ilist, title)
        
    elif(choice=="4"):
        '''
    Na série Chaos Noise seguirão as mesmas análises, usando as séries Logística
    e Henon, ambas vão gerar parâmetros aleatórios dentro de um intervalo de 
    rho=3.85~4 para a série logística, e a=1.35~1.4, b=0.21~0.29 para a série de
    Henon.
    Totalizando em 60 sinais no total.
        '''
        title="Série: Chaos Noise. {0}"
        d,ilist,rawdata=makeseries(Logistic, ["Logistic"], 30)
        aux1,aux2,aux3=makeseries(HenonMap, ["Henon"], 30)
        rawdata+=aux3
        d+=aux1
        ilist+=aux2
        for i in range(len(rawdata)):
            plt.figure(figsize=(20, 12))
            #Plot da série temporal
            ax1 = plt.subplot(211)
            ax1.set_title(title.format(rawdata[i][0]), fontsize=18)
            ax1.plot(rawdata[i][1],rawdata[i][2],color="firebrick", marker='o', linestyle='')
            #Plot e cálculo do DFA
            ax2 = plt.subplot(223)
            ax2.set_title(r"Detrended Fluctuation Analysis $\alpha$={0:.3}".format(rawdata[i][3], fontsize=15))
            ax2.plot(rawdata[i][4],rawdata[i][5], marker='o', linestyle='', color="#12355B")
            ax2.plot(rawdata[i][4], rawdata[i][6], color="#9DACB2")
            #Plot e cálculo do PSD
            ax3 = plt.subplot(224)
            ax3.set_title(r"Power Spectrum Density $\beta$={0:.3}".format(rawdata[i][12]), fontsize=15)
            ax3.set_yscale('log')
            ax3.set_xscale('log')
            ax3.plot(rawdata[i][7], rawdata[i][8], '-', color = 'deepskyblue', alpha = 0.7)
            ax3.plot(rawdata[i][9], rawdata[i][10], color = "darkblue", alpha = 0.8)
            ax3.axvline(rawdata[i][7][rawdata[i][14]], color = "darkblue", linestyle = '--')
            ax3.axvline(rawdata[i][7][rawdata[i][15]], color = "darkblue", linestyle = '--')    
            ax3.plot(rawdata[i][9], rawdata[i][13](rawdata[i][9], rawdata[i][11], rawdata[i][12]),color="#D65108", linestyle='-', linewidth = 3, label = '$%.4f$' %(rawdata[i][12]))
            ax2.set_xlabel("log(s)")
            ax2.set_ylabel("log F(s)")
            ax3.set_xlabel("Frequência (Hz)")
            ax3.set_ylabel("Potência")
            plt.savefig("Chaosserietemporalpsddfa{}.png".format(i))
            plt.show()
        title="chaosnoise"
        d=makeK(d,ilist, title)
    elif(choice=="5"):
        namefile=["sol3ghz.dat","surftemp504.txt", "covidbrasil.dat"]
        ind=int(input("1 sol3ghz, 2 - surftemp504, 3 - COVID Brasil\n\n"))
        title=namefile[ind-1]
        ilist=[title]
        fileread=open(namefile[ind-1])
        y=[]
        for line in fileread:
            y.append(float(line))
        alfa,xdfa,ydfa, reta = statsfuncs.dfa1d(y,1)
        freqs, power, xdata, ydata, amp, index, powerlaw, INICIO, FIM = statsfuncs.psd(y)
        psi=mfdfa.makemfdfa(y)
        d=pd.DataFrame({'Variance': statsfuncs.variance(y), 'Skewness': statsfuncs.skewness(y), 'Kurtosis': statsfuncs.kurtosis(y)+3, 'Alpha': alfa, 'Beta': index, "Psi": psi}, index=ilist)
        s2=d['Skewness']**2 #skew²
        k=d["Kurtosis"]
        alpha=d["Alpha"]
        beta=d["Beta"]
        legend=title
        cullenfrey(s2,k, legend, title) #Mapa de cullen e frey
        plt.title("EPSB-K-means, {}".format(title))
        
        return
    else:
        return
    #Fazendo o clustering, via kmeans, dos momentos estatísticos obtidos.
    ilist=set(ilist) #tirando valores duplicados
    for j in ilist:
        #Fazendo listas pra plotar em outras funções
        s2=[i**2 for i in d.filter(like=str(j), axis=0)['Skewness']] #skew²
        k=d.filter(like=str(j),axis=0)["Kurtosis"]
        alpha=d.filter(like=str(j),axis=0)["Alpha"]
        beta=d.filter(like=str(j),axis=0)["Beta"]
        legend=str(j)
        cullenfrey(s2,k, legend, title) #Mapa de cullen e frey
        makespaces(s2,k, alpha, beta, legend, title)
    return d

if __name__ == "__main__":
    d=main()
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(d)
