
from . import odds, probability
from .PMF import PMF
from .Hist import Hist

class Suite(PMF):
    """Represents a suite of hypotheses and their probabilities."""

    @classmethod
    def fromList(cls,t,name=""):
        hist = Hist.fromList(t)
        d = hist.getDict()
        return cls.fromDict(d)

    @classmethod
    def fromHist(cls,hist,name=None):
        if name is None:
            name = hist.name
        d = dict(hist.getDict())
        return cls.fromDict(d,name)

    @classmethod
    def fromDict(cls,d,name=""):
        suite = cls(name=name)
        suite.setDict(d)
        suite.normalize()
        return suite

    @classmethod
    def fromCDF(cls,cdf,name=None):
        if name is None:
            name = cdf.name
        suite = cls(name=name)
        prev = 0.0
        for v, p in cdf.items():
            suite.incr(v,p-prev)
            prev = p
        return suite



    def update(self, data):
        """Updates each hypothesis based on the data.

        data: any representation of the data

        returns: the normalizing constant
        """
        for hypo in list(self.values()):
            like = self.likelihood(data, hypo)
            self.mult(hypo, like)
        return self.normalize()

    def logUpdate(self, data):
        """Updates a suite of hypotheses based on new data.

        Modifies the suite directly; if you want to keep the original, make
        a copy.

        Note: unlike Update, LogUpdate does not normalize.

        Args:
            data: any representation of the data
        """
        for hypo in self.values():
            like = self.logLikelihood(data, hypo)
            self.incr(hypo, like)

    def updateSet(self, dataset):
        """Updates each hypothesis based on the dataset.

        This is more efficient than calling Update repeatedly because
        it waits until the end to Normalize.

        Modifies the suite directly; if you want to keep the original, make
        a copy.

        dataset: a sequence of data

        returns: the normalizing constant
        """
        for data in dataset:
            for hypo in self.values():
                like = self.likelihood(data, hypo)
                self.mult(hypo, like)
        return self.normalize()

    def logUpdateSet(self, dataset):
        """Updates each hypothesis based on the dataset.

        Modifies the suite directly; if you want to keep the original, make
        a copy.

        dataset: a sequence of data

        returns: None
        """
        for data in dataset:
            self.logUpdate(data)

    def likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        hypo: some representation of the hypothesis
        data: some representation of the data
        """
        raise NotImplementedError

    def logLikelihood(self, data, hypo):
        """Computes the log likelihood of the data under the hypothesis.

        hypo: some representation of the hypothesis
        data: some representation of the data
        """
        raise NotImplementedError

    def print(self):
        """Prints the hypotheses and their probabilities."""
        for hypo, prob in sorted(self.items()):
            print(hypo, prob)

    def makeOdds(self):
        """Transforms from probabilities to odds.

        Values with prob=0 are removed.
        """
        for hypo, prob in self.items():
            if prob:
                self.set(hypo, odds(prob))
            else:
                self.remove(hypo)

    def makeProbs(self):
        """Transforms from odds to probabilities."""
        for hypo, odds in self.items():
            self.set(hypo, probability(odds))