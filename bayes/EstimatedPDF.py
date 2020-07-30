
import scipy.stats

from .PDF import PDF
from .PMF import PMF

class EstimatedPDF(PDF):
    """Represents a PDF estimated by KDE."""

    def __init__(self, sample):
        """Estimates the density function based on a sample.

        sample: sequence of data
        """
        self.kde = scipy.stats.gaussian_kde(sample)

    def density(self, x):
        """Evaluates this Pdf at x.

        Returns: float probability density
        """
        return self.kde.evaluate(x)

    def makePmf(self, xs, name=''):
        ps = self.kde.evaluate(xs)
        pmf = PMF.fromItems(zip(xs, ps), name=name)
        return pmf

