
from . import evalGaussianPDF
from .PDF import PDF

class GaussianPDF(PDF):
    """Represents the PDF of a Gaussian distribution."""

    def __init__(self, mu, sigma):
        """Constructs a Gaussian Pdf with given mu and sigma.

        mu: mean
        sigma: standard deviation
        """
        self.mu = mu
        self.sigma = sigma

    def density(self, x):
        """Evaluates this Pdf at x.

        Returns: float probability density
        """
        return evalGaussianPDF(x, self.mu, self.sigma)