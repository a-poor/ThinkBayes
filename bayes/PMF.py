
import random
import logging

import numpy as np

from ._DictWrapper import _DictWrapper
from .Hist import Hist
from .CDF import CDF

class PMF(_DictWrapper):
    """Represents a probability mass function.
    
    Values can be any hashable type; probabilities are floating-point.
    Pmfs are not necessarily normalized.
    """

    @classmethod
    def fromList(cls,t,name=""):
        hist = Hist.fromList(t)
        d = hist.getDict()
        pmf = cls(d,name)
        pmf.normalize()
        return pmf

    @classmethod
    def fromDict(cls,d,name=""):
        pmf = cls(d,name)
        pmf.normalize()
        return pmf

    @classmethod
    def fromItems(cls,t,name=""):
        pmf = cls(dict(t),name)
        pmf.normalize()
        return pmf

    @classmethod
    def fromHist(cls,hist,name=None):
        if name is None:
            name = hist.name
        d = dict(hist.getDict())
        pmf = cls(d,name)
        pmf.normalize()
        return pmf

    @classmethod
    def fromCDF(cls,cdf,name=None):
        if name is None:
            name = cdf.name
        pmf = cls(name=name)
        prev = 0.0
        for v, p in cdf.items():
            pmf.incr(v,p-prev)
            prev = p
        return pmf

    @classmethod
    def makeMixture(cls,metapmf,name='mix'):
        mix = cls(name=name)
        for pmf, p1 in metapmf.items():
            for x, p2 in pmf.items():
                mix.incr(x,p1*p2)
        return mix

    @classmethod
    def makeUniformPMF(cls,low,high,n=None):
        if n is None:
            n = high - low + 1
        pmf = cls()
        for x in np.linspace(low,high,n):
            pmf.set(x,1)
        pmf.normalize()
        return pmf

    def prob(self, x, default=0):
        """Gets the probability associated with the value x.

        Args:
            x: number value
            default: value to return if the key is not there

        Returns:
            float probability
        """
        return self.d.get(x, default)

    def probs(self, xs):
        """Gets probabilities for a sequence of values."""
        return [self.prob(x) for x in xs]

    def makeCdf(self, name=None):
        """Makes a Cdf."""
        return MakeCdfFromPmf(self, name=name)

    def probGreater(self, x):
        """Probability that a sample from this Pmf exceeds x.

        x: number

        returns: float probability
        """
        t = [prob for (val, prob) in self.d.items() if val > x]
        return sum(t)

    def probLess(self, x):
        """Probability that a sample from this Pmf is less than x.

        x: number

        returns: float probability
        """
        t = [prob for (val, prob) in self.d.items() if val < x]
        return sum(t)

    def __lt__(self, obj):
        """Less than.

        obj: number or _DictWrapper

        returns: float probability
        """
        if isinstance(obj, _DictWrapper):
            return PmfProbLess(self, obj)
        else:
            return self.probLess(obj)

    def __gt__(self, obj):
        """Greater than.

        obj: number or _DictWrapper

        returns: float probability
        """
        if isinstance(obj, _DictWrapper):
            return PmfProbGreater(self, obj)
        else:
            return self.probGreater(obj)

    def __ge__(self, obj):
        """Greater than or equal.

        obj: number or _DictWrapper

        returns: float probability
        """
        return 1 - (self < obj)

    def __le__(self, obj):
        """Less than or equal.

        obj: number or _DictWrapper

        returns: float probability
        """
        return 1 - (self > obj)

    def __eq__(self, obj):
        """Equal to.

        obj: number or _DictWrapper

        returns: float probability
        """
        if isinstance(obj, _DictWrapper):
            return PmfProbEqual(self, obj)
        else:
            return self.prob(obj)

    def __ne__(self, obj):
        """Not equal to.

        obj: number or _DictWrapper

        returns: float probability
        """
        return 1 - (self == obj)

    def normalize(self, fraction=1.0):
        """Normalizes this PMF so the sum of all probs is fraction.

        Args:
            fraction: what the total should be after normalization

        Returns: the total probability before normalizing
        """
        if self.log:
            raise ValueError("Pmf is under a log transform")

        total = self.total()
        if total == 0.0:
            raise ValueError('total probability is zero.')
            logging.warning('Normalize: total probability is zero.')
            return total

        factor = float(fraction) / total
        for x in self.d:
            self.d[x] *= factor

        return total

    def random(self):
        """Chooses a random element from this PMF.

        Returns:
            float value from the Pmf
        """
        if len(self.d) == 0:
            raise ValueError('Pmf contains no values.')

        target = random.random()
        total = 0.0
        for x, p in self.d.items():
            total += p
            if total >= target:
                return x

        # we shouldn't get here
        assert False

    def mean(self):
        """Computes the mean of a PMF.

        Returns:
            float mean
        """
        mu = 0.0
        for x, p in self.d.items():
            mu += p * x
        return mu

    def var(self, mu=None):
        """Computes the variance of a PMF.

        Args:
            mu: the point around which the variance is computed;
                if omitted, computes the mean

        Returns:
            float variance
        """
        if mu is None:
            mu = self.mean()

        var = 0.0
        for x, p in self.d.items():
            var += p * (x - mu) ** 2
        return var

    def maximumLikelihood(self):
        """Returns the value with the highest probability.

        Returns: float probability
        """
        _, val = max((prob, val) for val, prob in self.items())
        return val

    def credibleInterval(self, percentage=90):
        """Computes the central credible interval.

        If percentage=90, computes the 90% CI.

        Args:
            percentage: float between 0 and 100

        Returns:
            sequence of two floats, low and high
        """
        cdf = self.makeCdf()
        return cdf.credibleInterval(percentage)

    def __add__(self, other):
        """Computes the Pmf of the sum of values drawn from self and other.

        other: another Pmf

        returns: new Pmf
        """
        try:
            return self.addPmf(other)
        except AttributeError:
            return self.addConstant(other)

    def addPmf(self, other):
        """Computes the Pmf of the sum of values drawn from self and other.

        other: another Pmf

        returns: new Pmf
        """
        pmf = Pmf()
        for v1, p1 in self.items():
            for v2, p2 in other.items():
                pmf.incr(v1 + v2, p1 * p2)
        return pmf

    def addConstant(self, other):
        """Computes the Pmf of the sum a constant and  values from self.

        other: a number

        returns: new Pmf
        """
        pmf = Pmf()
        for v1, p1 in self.items():
            pmf.set(v1 + other, p1)
        return pmf

    def __sub__(self, other):
        """Computes the Pmf of the diff of values drawn from self and other.

        other: another Pmf

        returns: new Pmf
        """
        pmf = Pmf()
        for v1, p1 in self.items():
            for v2, p2 in other.items():
                pmf.incr(v1 - v2, p1 * p2)
        return pmf

    def max(self, k):
        """Computes the CDF of the maximum of k selections from this dist.

        k: int

        returns: new Cdf
        """
        cdf = self.makeCdf()
        cdf.ps = [p ** k for p in cdf.ps]
        return cdf

    def __hash__(self):
        # FIXME
        # This imitates python2 implicit behaviour, which was removed in python3

        # Some problems with an id based hash:
        # looking up different pmfs with the same contents will give different values
        # looking up a new Pmf will always produce a keyerror

        # A solution might be to make a "FrozenPmf" immutable class (like frozenset)
        # and base a hash on a tuple of the items of self.d
        return id(self)