
# import bisect
import copy
# import logging
import math
# import numpy
# import random

# import scipy.stats
# from scipy.special import erf, erfinv, gammaln

class _DictWrapper:
    """An object that contains a dictionary."""

    def __init__(self, values=None, name=''):
        """Initializes the distribution.

        hypos: sequence of hypotheses
        """
        self.name = name
        self.d = {}

        # flag whether the distribution is under a log transform
        self.is_log = False

        if values is None:
            return

        init_methods = [
            self.initPmf,
            self.initMapping,
            self.initSequence,
            self.initFailure,
            ]

        for method in init_methods:
            try:
                method(values)
                break
            except AttributeError:
                continue

        if len(self) > 0:
            self.normalize()

    def initSequence(self, values):
        """Initializes with a sequence of equally-likely values.

        values: sequence of values
        """
        for value in values:
            self.set(value, 1)

    def initMapping(self, values):
        """Initializes with a map from value to probability.

        values: map from value to probability
        """
        for value, prob in values.items():
            self.set(value, prob)

    def initPmf(self, values):
        """Initializes with a Pmf.

        values: Pmf object
        """
        for value, prob in values.items():
            self.set(value, prob)

    def initFailure(self, values):
        """Raises an error."""
        raise ValueError('None of the initialization methods worked.')

    def __len__(self):
        return len(self.d)

    def __iter__(self):
        return iter(self.d)

    def keys(self):
        return iter(self.d)

    def __contains__(self, value):
        return value in self.d

    def copy(self, name=None):
        """Returns a copy.

        Make a shallow copy of d.  If you want a deep copy of d,
        use copy.deepcopy on the whole object.

        Args:
            name: string name for the new Hist
        """
        new = copy.copy(self)
        new.d = copy.copy(self.d)
        new.name = name if name is not None else self.name
        return new

    def scale(self, factor):
        """Multiplies the values by a factor.

        factor: what to multiply by

        Returns: new object
        """
        new = self.copy()
        new.d.clear()

        for val, prob in self.items():
            new.set(val * factor, prob)
        return new

    def normalize(self):
        raise NotImplementedError

    def log(self, m=None):
        """Log transforms the probabilities.
        
        Removes values with probability 0.

        Normalizes so that the largest logprob is 0.
        """
        if self.is_log:
            raise ValueError("Pmf/Hist already under a log transform")
        self.is_log = True

        if m is None:
            m = self.maxLike()

        for x, p in self.d.items():
            if p:
                self.set(x, math.log(p / m))
            else:
                self.remove(x)

    def exp(self, m=None):
        """Exponentiates the probabilities.

        m: how much to shift the ps before exponentiating

        If m is None, normalizes so that the largest prob is 1.
        """
        if not self.is_log:
            raise ValueError("Pmf/Hist not under a log transform")
        self.is_log = False

        if m is None:
            m = self.maxLike()

        for x, p in self.d.items():
            self.set(x, math.exp(p - m))

    def getDict(self):
        """Gets the dictionary."""
        return self.d

    def setDict(self, d):
        """Sets the dictionary."""
        self.d = d

    def values(self):
        """Gets an unsorted sequence of values.

        Note: one source of confusion is that the keys of this
        dictionary are the values of the Hist/Pmf, and the
        values of the dictionary are frequencies/probabilities.
        """
        return self.d.keys()

    def items(self):
        """Gets an unsorted sequence of (value, freq/prob) pairs."""
        return self.d.items()

    def render(self):
        """Generates a sequence of points suitable for plotting.

        Returns:
            tuple of (sorted value sequence, freq/prob sequence)
        """
        return zip(*sorted(self.items()))

    def print(self):
        """Prints the values and freqs/probs in ascending order."""
        str_items = [(str(v),str(p)) for v,p in sorted(self.items())]
        max_lens = [
            max(i[0] for i in str_items),
            max(i[1] for i in str_items)
        ]
        lena, lenb = max_lens
        print("\n".join(
            f"{v:{lena}s} -> {p:{lenb}s}"
            for v, p in str_items
        ))

    def set(self, x, y=0):
        """Sets the freq/prob associated with the value x.

        Args:
            x: number value
            y: number freq or prob
        """
        self.d[x] = y

    def incr(self, x, term=1):
        """Increments the freq/prob associated with the value x.

        Args:
            x: number value
            term: how much to increment by
        """
        self.d[x] = self.d.get(x, 0) + term

    def mult(self, x, factor):
        """Scales the freq/prob associated with the value x.

        Args:
            x: number value
            factor: how much to multiply by
        """
        self.d[x] = self.d.get(x, 0) * factor

    def remove(self, x):
        """Removes a value.

        Throws an exception if the value is not there.

        Args:
            x: value to remove
        """
        del self.d[x]

    def total(self):
        """Returns the total of the frequencies/probabilities in the map."""
        total = sum(self.d.values())
        return total

    def maxLike(self):
        """Returns the largest frequency/probability in the map."""
        return max(self.d.values())