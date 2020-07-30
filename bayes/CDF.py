
import random
import bisect

from ._DictWrapper import _DictWrapper
from .PMF import PMF
from .Hist import Hist

class CDF(_DictWrapper):
    """Represents a cumulative distribution function.

    Attributes:
        xs: sequence of values
        ps: sequence of probabilities
        name: string used as a graph label.
    """

    @classmethod
    def fromItems(cls,items,name=""):
        runsum = 0
        xs = []
        cs = []
        for v, c in sorted(items):
            runsum += c
            xs.append(v)
            cs.append(runsum)
        total = float(runsum)
        ps = [c / total for c in cs]
        return cls(xs,ps,name)

    @classmethod
    def fromDict(cls,d,name=""):
        return cls.fromItems(d.items(),name)

    @classmethod
    def fromHist(cls,hist,name=""):
        return cls.fromItems(hist.items(),name)

    @classmethod
    def fromPMF(cls,pmf,name=None):
        if name is None:
            name = pmf.name
        return cls.fromItems(pmf.items(),name)

    @classmethod
    def fromList(cls,seq,name=""):
        hist = Hist.fromList(seq)
        return cls.fromHist(hist,name)




    def __init__(self, xs=None, ps=None, name=''):
        self.xs = [] if xs is None else xs
        self.ps = [] if ps is None else ps
        self.name = name

    def copy(self, name=None):
        """Returns a copy of this Cdf.

        Args:
            name: string name for the new Cdf
        """
        if name is None:
            name = self.name
        return CDF(list(self.xs), list(self.ps), name)

    def makePMF(self, name=None):
        """Makes a Pmf."""
        return PMF.fromCDF(self, name=name)

    def values(self):
        """Returns a sorted list of values.
        """
        return self.xs

    def items(self):
        """Returns a sorted sequence of (value, probability) pairs.

        Note: in Python3, returns an iterator.
        """
        return zip(self.xs, self.ps)

    def append(self, x, p):
        """Add an (x, p) pair to the end of this CDF.

        Note: this us normally used to build a CDF from scratch, not
        to modify existing CDFs.  It is up to the caller to make sure
        that the result is a legal CDF.
        """
        self.xs.append(x)
        self.ps.append(p)

    def shift(self, term):
        """Adds a term to the xs.

        term: how much to add
        """
        new = self.copy()
        new.xs = [x + term for x in self.xs]
        return new

    def scale(self, factor):
        """Multiplies the xs by a factor.

        factor: what to multiply by
        """
        new = self.copy()
        new.xs = [x * factor for x in self.xs]
        return new

    def prob(self, x):
        """Returns CDF(x), the probability that corresponds to value x.

        Args:
            x: number

        Returns:
            float probability
        """
        if x < self.xs[0]: return 0.0
        index = bisect.bisect(self.xs, x)
        p = self.ps[index - 1]
        return p

    def value(self, p):
        """Returns InverseCDF(p), the value that corresponds to probability p.

        Args:
            p: number in the range [0, 1]

        Returns:
            number value
        """
        if p < 0 or p > 1:
            raise ValueError('Probability p must be in range [0, 1]')

        if p == 0: return self.xs[0]
        if p == 1: return self.xs[-1]
        index = bisect.bisect(self.ps, p)
        if p == self.ps[index - 1]:
            return self.xs[index - 1]
        else:
            return self.xs[index]

    def percentile(self, p):
        """Returns the value that corresponds to percentile p.

        Args:
            p: number in the range [0, 100]

        Returns:
            number value
        """
        return self.value(p / 100.0)

    def random(self):
        """Chooses a random value from this distribution."""
        return self.value(random.random())

    def sample(self, n):
        """Generates a random sample from this distribution.
        
        Args:
            n: int length of the sample
        """
        return [self.random() for i in range(n)]

    def mean(self):
        """Computes the mean of a CDF.

        Returns:
            float mean
        """
        old_p = 0
        total = 0.0
        for x, new_p in zip(self.xs, self.ps):
            p = new_p - old_p
            total += p * x
            old_p = new_p
        return total

    def credibleInterval(self, percentage=90):
        """Computes the central credible interval.

        If percentage=90, computes the 90% CI.

        Args:
            percentage: float between 0 and 100

        Returns:
            sequence of two floats, low and high
        """
        prob = (1 - percentage / 100.0) / 2
        interval = self.value(prob), self.value(1 - prob)
        return interval

    def _round(self, multiplier=1000.0):
        """
        An entry is added to the cdf only if the percentile differs
        from the previous value in a significant digit, where the number
        of significant digits is determined by multiplier.  The
        default is 1000, which keeps log10(1000) = 3 significant digits.
        """
        # TODO(write this method)
        raise NotImplementedError

    def render(self):
        """Generates a sequence of points suitable for plotting.

        An empirical CDF is a step function; linear interpolation
        can be misleading.

        Returns:
            tuple of (xs, ps)
        """
        xs = [self.xs[0]]
        ps = [0.0]
        for i, p in enumerate(self.ps):
            xs.append(self.xs[i])
            ps.append(p)

            try:
                xs.append(self.xs[i + 1])
                ps.append(p)
            except IndexError:
                pass
        return xs, ps

    def max(self, k):
        """Computes the CDF of the maximum of k selections from this dist.

        k: int

        returns: new Cdf
        """
        cdf = self.copy()
        cdf.ps = [p ** k for p in cdf.ps]
        return cdf