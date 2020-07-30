
from .PMF import PMF

class Joint(PMF):
    """Represents a joint distribution.

    The values are sequences (usually tuples)
    """

    def marginal(self, i, name=''):
        """Gets the marginal distribution of the indicated variable.

        i: index of the variable we want

        Returns: Pmf
        """
        pmf = PMF(name=name)
        for vs, prob in self.items():
            pmf.incr(vs[i], prob)
        return pmf

    def conditional(self, i, j, val, name=''):
        """Gets the conditional distribution of the indicated variable.

        Distribution of vs[i], conditioned on vs[j] = val.

        i: index of the variable we want
        j: which variable is conditioned on
        val: the value the jth variable has to have

        Returns: Pmf
        """
        pmf = PMF(name=name)
        for vs, prob in self.items():
            if vs[j] != val: continue
            pmf.incr(vs[i], prob)

        pmf.normalize()
        return pmf

    def maxLikeInterval(self, percentage=90):
        """Returns the maximum-likelihood credible interval.

        If percentage=90, computes a 90% CI containing the values
        with the highest likelihoods.

        percentage: float between 0 and 100

        Returns: list of values from the suite
        """
        interval = []
        total = 0

        t = [(prob, val) for val, prob in self.items()]
        t.sort(reverse=True)

        for prob, val in t:
            interval.append(val)
            total += prob
            if total >= percentage / 100.0:
                break

        return interval
