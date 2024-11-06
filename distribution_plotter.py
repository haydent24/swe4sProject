import pandas as pd
import numpy as np

"""
Contains functions to generate and plot datasets that fall into different distribution types
"""

def norm_gen():
    mean = 0       # mean of the distribution
    std_dev = 1    # standard deviation of the distribution

    # Generate 10,000 data points
    data = np.random.normal(loc=mean, scale=std_dev, size=10000)
    # Convert to pandas dataframe
    df = pd.DataFrame(data, columns=["Data Points"])
    
    #plotting section
    
def poisson_gen():
    lam = 5 # events per interval
    
    data = np.random.poisson(lam=lam, size=10000)
    df = pd.DataFrame(data, columns = ["Data Points"])
    
    #plotting section
    
def exp_gen():
    time = 1.0
    
    data = np.random.exponential(scale=time, size=10000)
    df = pd.DataFrame(data, columns = ["Data Points"])
    
    #plotting section
    
def chi_gen():
    degrees = 2
    
    data = np.random.chisquare(df=degrees, size=10000)
    df = pd.DataFrame(data, columns = ["Data Points"])
    
    #plotting section
    
def beta_gen():
    alpha = 2
    beta = 5
    
    data = np.random.beta(a=alpha, b=beta, size=10000)
    df = pd.DataFrame(data, columns = ["Data Points"])
    
    #plotting section

def gamma_gen():
    shape = 2
    scale = 1.0
    
    data = np.random.gamma(shape=shape, scale=scale, size=10000)
    df = pd.DataFrame(data, columns = ["Data Points"])
    
    #plotting section
    
def log_norm_gen():
    mean = 0
    std_dev = 1
    
    data = np.random.lognormal(mean=mean, sigma=std_dev, size=10000)
    df = pd.DataFrame(data, columns = ["Data Points"])
    
    #plotting section

def weibull_gen():
    shape = 1.5
    
    data = np.random.weibull(a=shape, size=10000)
    df = pd.DataFrame(data, columns = ["Data Points"])
    
    #plotting section
