
import math
import random

import numpy as np
import scipy.stats
from scipy.special import erf, erfinv, gammaln

from .Hist import Hist
from .Joint import Joint
from .PMF import PMF
from .CDF import CDF
from .PDF import PDF
from .Suite import Suite
from .GaussianPDF import GaussianPDF
from .EstimatedPDF import EstimatedPDF
from .Beta import Beta
from .Dirichlet import Dirichlet


ROOT2 = 2 ** 0.5

def seed(n: int) -> None:
    random.seed(n)
    np.random.seed(n)

def odds(p: float) -> float:
    if p == 1:
        return float('inf')
    else:
        return p / (1 - p)

def probability(o: float) -> float:
    return o / (o + 1)


def makeJoint(pmf1: PMF, pmf2: PMF) -> Joint:
    joint = Joint()
    for v1, p1 in pmf1.items():
        for v2, p2 in pmf2.items():
            joint.set((v1,v2), p1 * p2)
    return joint

def makeHistFromList(t: list, name="") -> Hist:
    return Hist.fromList(t,name)

def makeHistFromDict(d: dict, name="") -> Hist:
    return Hist.fromDict(d,name)

def makePmfFromList(t: list, name="") -> PMF:
    return PMF.fromList(t,name)

def makePmfFromDict(d: dict, name="") -> PMF:
    return PMF.fromDict(d,name)

def makePmfFromItems(t, name="") -> PMF:
    return PMF.fromItems(t,name)

def makeMixture(metapmf,name='mix'):
    return PMF.makeMixture(metapmf,name)


def makeCdfFromItems(items, name=''):
    """Makes a cdf from an unsorted sequence of (value, frequency) pairs.

    Args:
        items: unsorted sequence of (value, frequency) pairs
        name: string name for this CDF

    Returns:
        cdf: list of (value, fraction) pairs
    """
    return CDF.fromItems(items,name)


def makeCdfFromDict(d, name=''):
    """Makes a CDF from a dictionary that maps values to frequencies.

    Args:
       d: dictionary that maps values to frequencies.
       name: string name for the data.

    Returns:
        Cdf object
    """
    return CDF.fromDict(d,name)


def makeCdfFromHist(hist, name=''):
    """Makes a CDF from a Hist object.

    Args:
       hist: Pmf.Hist object
       name: string name for the data.

    Returns:
        Cdf object
    """
    return CDF.fromHist(hist,name)


def makeCdfFromPmf(pmf, name=None):
    """Makes a CDF from a Pmf object.

    Args:
       pmf: Pmf.Pmf object
       name: string name for the data.

    Returns:
        Cdf object
    """
    return CDF.fromPMF(pmf,name)


def makeCdfFromList(seq, name=''):
    """Creates a CDF from an unsorted sequence.

    Args:
        seq: unsorted sequence of sortable values
        name: string name for the cdf

    Returns:
       Cdf object
    """
    return CDF.fromList(seq, name)

def percentile(pmf,percentage):
    p = percentage / 100.0
    total = 0
    for val, prob in pmf.items():
        total += prob
        if total >= p:
            return val

def credibleInterval(pmf,percentage=90):
    cdf = pmf.makeCDF()
    prob = (1-percentage/100.0) / 2
    interval = cdf.value(prob), cdf.value(1-prob)
    return interval

def pmfProbLess(pmf1,pmf2):
    total = 0.0
    for v1, p1, in pmf1.items():
        for v2, p2 in pmf2.items():
            if v1 < v2:
                total += p1 * p2
    return total

def pmfProbGreater(pmf1,pmf2):
    total = 0.0
    for v1, p1, in pmf1.items():
        for v2, p2 in pmf2.items():
            if v1 > v2:
                total += p1 * p2
    return total

def pmfProbEqual(pmf1,pmf2):
    total = 0.0
    for v1, p1, in pmf1.items():
        for v2, p2 in pmf2.items():
            if v1 == v2:
                total += p1 * p2
    return total

def randomSum(dists):
    return sum(dist.random() for dist in dists)

def sampleSum(dists,n):
    return PMF.fromList(randomSum(dists) for _ in range(n))

def evalGaussianPDF(x,mu,sigma):
    return scipy.stats.norm.pdf(x,mu,sigma)

def makeGaussianPMF(mu,sigma,num_sigmas,n=201):
    """Makes a PMF discrete approx to a Gaussian distribution.
    
    mu: float mean
    sigma: float standard deviation
    num_sigmas: how many sigmas to extend in each direction
    n: number of values in the Pmf

    returns: normalized Pmf
    """
    pmf = PMF()
    low = mu - num_sigmas * sigma
    high = mu + num_sigmas * sigma
    for x in np.linspace(low,high,n):
        p = evalGaussianPDF(x,mu,sigma)
        pmf.set(x,p)
    pmf.normalize()
    return pmf

def evalBinomialPMF(k,n,p):
    """Evaluates the binomial pmf.

    Returns the probabily of k successes in n trials with probability p.
    """
    return scipy.stats.binom.pmf(k, n, p)

def evalPoissonPMF(k, lam):
    """Computes the Poisson PMF.

    k: number of events
    lam: parameter lambda in events per unit time

    returns: float probability
    """
    return scipy.stats.poisson.pmf(k, lam)

def makePoissonPMF(lam, high, step=1):
    """Makes a PMF discrete approx to a Poisson distribution.

    lam: parameter lambda in events per unit time
    high: upper bound of the Pmf

    returns: normalized Pmf
    """
    pmf = PMF()
    for k in range(0, high + 1, step):
        p = evalPoissonPMF(k, lam)
        pmf.set(k, p)
    pmf.normalize()
    return pmf

def evalExponentialPDF(x,lam):
    """Computes the exponential PDF.

    x: value
    lam: parameter lambda in events per unit time

    returns: float probability density
    """
    return lam * math.exp(-lam * x)

def evalExponentialCDF(x, lam):
    """Evaluates CDF of the exponential distribution with parameter lam."""
    return 1 - math.exp(-lam * x)

def makeExponentialPMF(lam, high, n=200):
    """Makes a PMF discrete approx to an exponential distribution.

    lam: parameter lambda in events per unit time
    high: upper bound
    n: number of values in the Pmf

    returns: normalized Pmf
    """
    pmf = PMF()
    for x in np.linspace(0, high, n):
        p = evalExponentialPDF(x, lam)
        pmf.set(x, p)
    pmf.normalize()
    return pmf

def standardGaussianCDF(x):
    """Evaluates the CDF of the standard Gaussian distribution.
    
    See http://en.wikipedia.org/wiki/Normal_distribution
    #Cumulative_distribution_function

    Args:
        x: float
                
    Returns:
        float
    """
    return (erf(x / ROOT2) + 1) / 2

def gaussianCDF(x, mu=0, sigma=1):
    """Evaluates the CDF of the gaussian distribution.
    
    Args:
        x: float

        mu: mean parameter
        
        sigma: standard deviation parameter
                
    Returns:
        float
    """
    return standardGaussianCDF(float(x - mu) / sigma)


def gaussianCdfInverse(p, mu=0, sigma=1):
    """Evaluates the inverse CDF of the gaussian distribution.

    See http://en.wikipedia.org/wiki/Normal_distribution#Quantile_function  

    Args:
        p: float

        mu: mean parameter
        
        sigma: standard deviation parameter
                
    Returns:
        float
    """
    x = ROOT2 * erfinv(2 * p - 1)
    return mu + x * sigma


def binomialCoef(n, k):
    """Compute the binomial coefficient "n choose k".

    n: number of trials
    k: number of successes

    Returns: float
    """
    return scipy.special.binom(n, k)


def logBinomialCoef(n, k):
    """Computes the log of the binomial coefficient.

    http://math.stackexchange.com/questions/64716/
    approximating-the-logarithm-of-the-binomial-coefficient

    n: number of trials
    k: number of successes

    Returns: float
    """
    return n * math.log(n) - k * math.log(k) - (n - k) * math.log(n - k)

