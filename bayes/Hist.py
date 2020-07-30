
from ._DictWrapper import _DictWrapper

class Hist(_DictWrapper):
    """Represents a histogram, which is a map from values to frequencies.

    Values can be any hashable type; frequencies are integer counters.
    """

    @classmethod
    def fromList(cls,t,name=""):
        hist = cls(name=name)
        [hist.incr(x) for x in t]
        return hist

    @classmethod
    def fromDict(cls,d,name=""):
        return cls(d,name)

    def freq(self, x):
        """Gets the frequency associated with the value x.

        Args:
            x: number value

        Returns:
            int frequency
        """
        return self.d.get(x, 0)

    def freqs(self, xs):
        """Gets frequencies for a sequence of values."""
        return [self.freq(x) for x in xs]

    def isSubset(self, other):
        """Checks whether the values in this histogram are a subset of
        the values in the given histogram."""
        for val, freq in self.items():
            if freq > other.freq(val):
                return False
        return True

    def subtract(self, other):
        """Subtracts the values in the given histogram from this histogram."""
        for val, freq in other.items():
            self.incr(val, -freq)