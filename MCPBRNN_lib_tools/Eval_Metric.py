import torch
import torch.nn as nn
import numpy as np

def correlation(s, o):
    """
        correlation coefficient
        input:
        s: simulated
        o: observed
        output:
        correlation: correlation coefficient
        """
    # s,o = filter_nan(s,o)
    if s.size == 0:
        corr = np.NaN
    else:
        corr = np.corrcoef(o.flatten(), s.flatten())[0,1]
                    
    return corr

def NS(s, o):
    """
        Nash Sutcliffe efficiency coefficient
        input:
        s: simulated
        o: observed
        output:
        ns: Nash Sutcliffe efficient coefficient
        """
    # s,o = filter_nan(s,o)
    return 1 - np.sum((s-o)**2)/np.sum((o-np.mean(o))**2)

def KGE(s, o):
    """
        Kling-Gupta Efficiency
        input:
        s: simulated
        o: observed
        output:
        kge: Kling-Gupta Efficiency
        cc: correlation
        alpha: ratio of the standard deviation
        beta: ratio of the mean
        """
    # s,o = filter_nan(s,o)
    cc = correlation(s,o)
    alpha = np.std(s)/np.std(o)
    beta = np.sum(s)/np.sum(o)
    kge = 1- np.sqrt( (cc-1)**2 + (alpha-1)**2 + (beta-1)**2 )
    kgess = 1-(1-kge)/np.sqrt(2)
    return kge, cc, alpha, beta, kgess