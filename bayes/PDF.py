
from .PMF import PMF

class PDF:
    """Represents a probability density function (PDF)."""

    def density(self, x):
        """Evaluates this Pdf at x.

        Returns: float probability density
        """
        raise NotImplementedError

    def makePmf(self, xs, name=''):
        """Makes a discrete version of this Pdf, evaluated at xs.

        xs: equally-spaced sequence of values

        Returns: new Pmf
        """
        pmf = PMF(name=name)
        for x in xs:
            pmf.set(x, self.density(x))
        pmf.normalize()
        return pmf