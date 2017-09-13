# -*- coding: utf-8 -*-
"""
Created on Fri Sep 01 09:28:08 2017

@author: Stefan Bonchev

StefanBonchev_FinalProject_010917

Code uses Python 2.7

Final Project: The Misbehaviour of Markets
1.	Write a python program(s) to download end-of-day data last 25 years the major global stock market indices from Google Finance, Yahoo Finance, Quandl, CityFALCON, or another similar source.
2.	It is a common assumption in quantitative finance that stock returns follow a normal distribution whereas prices follow a lognormal distribution For all these indices check how closely price movements followed a log-normal distribution.
3.	Verify whether returns from these broad market indices followed a normal distribution?
4.	For each of the above two parameters (price movements and stock returns) come up with specific statistical measures that clearly identify the degree of deviation from the ideal distributions. Graphically represent the degree of correspondence.
5.	One of the most notable hypothesis about stock market behavior is the “Efficient market hypothesis” which also internally assume that market price follows a random-walk process. Assuming that Stock Index prices follow a geometric Brownian motion and hence index returns were normally distributed with about 20% historical volatility, write a program sub-module to calculate the probability of an event like the 1987 stock market crash happening ? Explain in simple terms what the results imply.
6.	What does "fat tail" mean? Plot the distribution of price movements for the downloaded indices (in separate subplot panes of a graph) and identify fat tail locations if any.
7.	It is often claimed that fractals and multi-fractals generate a more realistic picture of market risks than log-normal distribution. Considering last 10 year daily price movements of NASDAQ, write a program to check whether fractal geometrics could have better predicted stock market movements than log-normal distribution assumption. Explain your findings with suitable graphs.

Indexes to be used:
    - S&P 500 (USA)
    - DAX (Germany)
    - FTSE (UK)
    - KOSPI (Korea)
"""



#Importing Necessary Modules
import fix_yahoo_finance as yf
import numpy as np
import scipy.stats as stats
from scipy.stats import kurtosis, skew
import pylab
import matplotlib.pyplot as plt
import pandas as pd 

#[1]Fetching end of day data for the last 25 years for the four indexes in play
"""This module downloads Market Indexes data from Yahoo Finance. For the purpose of our project we are
going to use S&P 500 (^GSPC), DAX (^GDAXI) and 
Footsie 100 (^FTSE)."""


SP500 = yf.download('^GSPC', start="1991-08-01", end="2017-08-01")
DAX = yf.download('^GDAXI', start="1991-08-01", end="2017-08-01")
FTSE = yf.download('^FTSE', start="1991-08-01", end="2017-08-01")
KOSPI = yf.download('^KOSPI', start="1991-08-01", end="2017-08-01")

#[2] Using a Normality test for the prices of the stock market indexes to find out their distribution
"""When we look at the descriptive statistics of the index prices when can make a judgement
 on the basis of their histograms, skewness, kurtosis and variance whether their distribution is log normal or normal"""

#calculating the descriptive stats for the indexes
scipy.stats.mstats.normaltest(SP500, DAX, FTSE, KOSPI, axis=0)
skew = scipy.stats.skewtest(SP500, DAX, FTSE, KOSPI, axis=0)
kurtosis = scipy.stats.kurtosistest(SP500, DAX, FTSE, KOSPI, axis=0)

print("mean : ", np.mean(SP500, DAX, FTSE, KOSPI))
print("var  : ", np.var(SP500, DAX, FTSE, KOSPI))
print("skew : ", skew(SP500, DAX, FTSE, KOSPI))
print("kurt : ", kurtosis(SP500, DAX, FTSE, KOSPI))

#Plotting histograms with the prices of the indexes
""" Creating a reusable function for plotting histograms with the MatplotLib Package for python"""

def index_histogram(index, bins=10, rot=90, align='center', color='green', title='')

   f, (ax1, ax2) = plt.subplots(1, 2)
   
   ax1.hist(index, bins='auto')
   ax1.set_title('probability density')
   ax2.hist(y, bins='auto')
   ax2.set_title('Indexes Histogram')

   ax.grid(False)

   plt.legend()
   plt.tight_layout()
   plt.show()

index_histogram(SP500, color='red', title='Histogram for the Prices of S&P500')
index_histogram(DAX, color='green', title='Histogram for the Prices of DAX')
index_histogram(FTSE, color='yellow', title='Histogram for the Prices of FTSE')
index_histogram(KOSPI, color='blue', title='Histogram for the Prices of KOSPI')

#Additionaly we are going to plot Q-Q Plots for the indexes
"""Looking at the Q-Q Plots is going to give us an idea whether the sets of data come from a common distribution"""

def index_QQ(index, rot=90, align='center', color='green', title=''):    
    
    stats.probplot(index, dist="norm", plot=pylab)
    
    pylab.legend()
    pylab.tight_layout()
    pylab.show()

index_QQ(SP500, color='red', title='QQ Plot for the Prices of S&P500')
index_QQ(DAX, color='green', title='QQ Plot for the Prices of S&P500')
index_QQ(FTSE, color='yellow', title='QQ Plot for the Prices of S&P500')
index_QQ(KOSPI, color='blue', title='QQ Plot for the Prices of S&P500')


#[3]Calculation of Returns for the given indexes and looking at their descriptive stats and histgrams
""" Calculating the yearly returns from 1920 onwards using the numpy library."""

if __Returns__ == '__Returns__':
    def calcReturn(ticker,begdate,enddate):
        SP500 = yf.download('^GSPC', start="1991-08-01", end="2017-08-01", asobject=True, adjusted=True)
        DAX = yf.download('^GDAXI', start="1991-08-01", end="2017-08-01", asobject=True, adjusted=True)
        FTSE = yf.download('^FTSE', start="1991-08-01", end="2017-08-01", asobject=True, adjusted=True)
        KOSPI = yf.download('^KOSPI', start="1991-08-01", end="2017-08-01", asobject=True, adjusted=True)
    
    begdate = (1991,8,1)
    enddate=(2017,8,1)
    SP500Return = calcReturn('^GSPC',begdate,enddate)
    retSP=calcReturn('^GSPC',begdate,enddate)
    n=min(len(returnSP500),len(retSP))
    s=np.ones(n)*2
    t=range(n)
        
    begdate = (1991,8,1)
    enddate=(2017,8,1)
    DAXreturn =calcReturn('^GDAXI',begdate,enddate)
    retDJI=calcReturn('^GDAXI',begdate,enddate)
    m =min(len(returnDJI),len(retDJI))
    s1=np.ones(m)*2
    t1=range(m)
    
    begdate = (1991,8,1)
    enddate=(2017,8,1)
    FTSEreturn = calcReturn('^FTSE',begdate,enddate)
    retDJI=calcReturn('^FTSE',begdate,enddate)
    m =min(len(returnDJI),len(retDJI))
    s1=np.ones(m)*2
    t1=range(m)
    
    begdate = (1991,8,1)
    enddate=(2017,8,1)
    KOSPIreturn = calcReturn('^KOSPI',begdate,enddate)
    retDJI=calcReturn('^KOSPI',begdate,enddate)
    m =min(len(returnDJI),len(retDJI))
    s1=np.ones(m)*2
    t1=range(m)

#calculating the descriptive stats for the return on the indexes
scipy.stats.mstats.normaltest(SP500return, DAXreturn, FTSEreturn, KOSPIreturn, axis=0)
skewR = scipy.stats.skewtest(SP500return, DAXreturn, FTSEreturn, KOSPIreturn, axis=0)
kurtosis4 = scipy.stats.kurtosistest(SP500return, DAXreturn, FTSEreturn, KOSPIreturn, axis=0)

print("SD : ", np.std(SP500return, DAXreturn, FTSEreturn, KOSPIreturn))
print("mean : ", np.mean(SP500return, DAXreturn, FTSEreturn, KOSPIreturn))
print("var  : ", np.var(SP500return, DAXreturn, FTSEreturn, KOSPIreturn))
print("skew : ",skewR(SP500return, DAXreturn, FTSEreturn, KOSPIreturn))
print("kurt : ",kurtosisR(SP500return, DAXreturn, FTSEreturn, KOSPIreturn))

#Plotting Histograms and QQ plots for the return on the indexes
#Histograms
index_histogram(SP500return, color='red', title='Histogram for the Prices of S&P500')
index_histogram(DAXreturn, color='green', title='Histogram for the Prices of DAX')
index_histogram(FTSEreturn, color='yellow', title='Histogram for the Prices of FTSE')
index_histogram(KOSPIreturn, color='blue', title='Histogram for the Prices of KOSPI')

#QQs
index_QQ(SP500return, color='red', title='QQ Plot for the Prices of S&P500')
index_QQ(DAXreturn, color='green', title='QQ Plot for the Prices of S&P500')
index_QQ(FTSEreturn, color='yellow', title='QQ Plot for the Prices of S&P500')
index_QQ(KOSPIreturn, color='blue', title='QQ Plot for the Prices of S&P500')

#Geometric Brownian motion and stock crash probability

mu=1
n=50
dt=0.1
x0=100
x=pd.DataFrame(SP500return, DAXreturn, FTSEreturn, KOSPIreturn)
np.random.seed(10000)

for sigma in np.arange(0.8,2,0.2):
    step=np.exp((mu-sigma**2/2)*dt)*np.exp(sigma*np.random.normal(0,dt,(1,n)))
    temp=pd.DataFrame(x0*step.cumprod())
    x=pd.concat([x,temp],axis=1)

x.columns=np.arange(0.8,2,0.2)
plt.plot(x)
plt.legend(x.columns)
plt.xlabel('t')
plt.ylabel('X')
plt.title('Realizations of Geometric Brownian motion with difference variances\n mu=1')
plt.show()

#Calculating and plotting fractals for the last 10 years for NASDAQ 
""" Using the Mandelbrotian Fractals analysis on the NASDAQ Index data to test for unsual risk levels"""

#Fetching the Market Index data for NASDAQ from Yahoo Finance
NASDAQ = yf.download('^IXIC', start="2007-08-01", end="2017-08-01")


def NASDAQ_Frac(C, NASDAQ):
    X = C
    for n in range(max_iter):
    if abs(X) > 2.:
    return n
    X = X ** 2 + C
    return max_iter
    N = 512
    max_iter = 64
    xmin, xmax, ymin, ymax = -2.2, .8, -1.5, 1.5
    X = np.linspace(xmin, xmax, N)
    Y = np.linspace(ymin, ymax, N)
    Z = np.empty((N, N))
    for i, y in enumerate(Y):
    for j, x in enumerate(X):
    Z[i, j] = iter_count(complex(x, y), max_iter)
    plt.imshow(Z, cmap = cm.gray)
    plt.show()