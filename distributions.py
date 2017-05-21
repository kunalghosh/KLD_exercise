import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

class distribution():
    def __init__(self,**kwargs):
        self.seed = kwargs['seed']
        np.random.seed(seed=self.seed)
        self.name = "distribution_{}".format(kwargs['name'])

    def sample(self,nsamples):
        pass
    def update_params(self,**kwargs):
        # I don't use self.__dict__.update()
        # because, this way 
        # someone could possibly add a new
        # attribute to the class
        # self.__dict__.update(kwargs)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    # class method allows to get subclass
    # name in superclass
    def plot(self,get_fig=False):
        samples = self.sample(nsamples=500)
        ax = sns.distplot(samples, hist=False, rug=True)
        # sns.plt.title("Distribution : %s" % self.name)
        fig = ax.get_figure()
        fig.savefig("%s.png" % self.name)
        if not get_fig:
            plt.close()
        else:
            return fig,plt

    def pdf(self):
        pass

    def logpdf(self):
        pass

    def norm_pdf(self, x, mu=0, sigma=1):
        # only for 1d
        val = np.exp(-0.5 * ((x - mu)/sigma)**2) / (np.sqrt(2*np.pi)*sigma)
        return val

    def norm_logpdf(self, x, mu=0, sigma=1):
        # only for 1d
        val = -0.5 * ( log(2*pi) + 2*log(sigma) + ((x-mu)/sigma)**2 )
        return val


class px(distribution):
    def __init__(self,name="mix_gauss",seed=1,nsamples=1,m1=0,m2=4,sigma1=1,
            sigma2=1,pi=0.5):
        super().__init__(name=name,seed=seed,nsamples=nsamples,m1=m1,m2=m2,
                        sigma1=sigma1,sigma2=sigma2,pi=pi)

        self.nsamples=nsamples
        self.m1=m1
        self.m2=m2
        self.sigma1=sigma1
        self.sigma2=sigma2
        self.pi=pi
    
    def sample(self, nsamples=None):
        num_samples = self.nsamples if nsamples is None else nsamples
        return p_x(num_samples, m1=self.m1, m2=self.m2, sigma1=self.sigma1,
                    sigma2=self.sigma2, pi=self.pi)

    # def evaluate(self,samples):
    #     # probabilities = self.pi*norm.pdf(samples ,loc=self.m1, scale=self.sigma1) + \
    #     #                (1-self.pi)*norm.pdf(samples, loc=self.m2, scale=self.sigma2)
    #     probabilities = self.pdf(samples)
    #     return probabilities

    def pdf(self, samples):
        probabilities = self.pi*self.norm_pdf(samples ,mu=self.m1, sigma=self.sigma1) + \
                        (1-self.pi)*self.norm_pdf(samples, mu=self.m2, sigma=self.sigma2)
        return probabilities

    def logpdf(self, samples):
        """
        Returns the logpdf of the mixture of gaussians.

        .. math ::
            log(\pi*N(x|\mu_1, \sigma_1)+(1-\pi)*N(x|\mu_2, \sigma_2)) =\\
                log(\pi) + log(N(x|\mu_1, \sigma_1)) + \\
                log(1 + ((1-\pi)/\pi) * (\sigma2 / \sigma_1) * \\
                \exp(0.5 * ( \\
                ((x-\mu_2)/\sigma_2)^2 - ((x-\mu_1)/\sigma_1)^2 \\
                )) \\
                )

        """
        pi,m1,m2,sigma1,sigma2 = self.pi,self.mu1,self.mu2,self.sigma1,self.sigma2
        probabilities = log(pi) + self.norm_logpdf(samples, m1, sigma1) + \
                        log( \
                                1 + ((1-pi)*sigma2)/(pi*sigma1) *  \
                                    exp(0.5 * ( \
                                        ((samples-m2)/sigma2)**2 - \
                                        ((samples-m1)/sigma1)**2 \
                                        ) \
                                ))
        return probabilities
    
class qx(distribution):
    def __init__(self,name="gauss",seed=1,nsamples=1,mu=0,sigma=1):
        super().__init__(name=name,seed=seed,nsamples=nsamples,mu=mu,
                sigma=sigma)

        self.nsamples=nsamples
        self.mu = mu
        self.sigma = sigma

    def sample(self, nsamples=None):
        num_samples = self.nsamples if nsamples is None else nsamples
        return q_x(num_samples, mu=self.mu, sigma=self.sigma)

    # def evaluate(self, samples):
    #     """
    #     Evaluates the probability of the samples originating from this distribution
    #     """
    #     # probabilities = norm.pdf(samples, loc=self.mu, scale=self.sigma)
    #     probabilities = self.pdf(samples, loc=self.mu, scale=self.sigma)
    #     return probabilities
    
    def pdf(self, samples):
        return self.norm_pdf(samples, mu=self.mu, sigma=self.sigma)

    def logpdf(self, samples):
        return self.norm_logpdf(samples, mu=self.mu, sigma=self.sigma)


def p_x(nsamples=1,m1=0,m2=4,sigma1=1,sigma2=1,pi=0.5):
    """
    The distribution to be approximated

    .. math ::
        x \sim \pi*N(\mu_1, \sigma_1) + (1-\pi) * N(\mu_2, \sigma_2)
    """
    samples = np.random.randn(nsamples,1)
    samples_uniform = np.random.rand(nsamples,1)
    for idx, sample_uniform in enumerate(samples_uniform):
        if sample_uniform < pi:
            samples[idx] = samples[idx] * sigma1 + m1
        else:
            samples[idx] = samples[idx] * sigma2 + m2
    return samples

def q_x(nsamples=1, mu=0, sigma=1):
    """
    The approximating distribution.

    .. math ::
        x \sim N(mu, sigma)
    """
    samples = np.random.randn(nsamples, 1) * sigma + mu
    return samples

if __name__ == "__main__":
    seed = 1
    np.random.seed(seed)
    assert seed == 1, "Seed is not 1. Tests will fail !"
    px = p_x()
    qx = q_x()
    assert px - 1.62434536 < 10e-5, "p_x doesn't implement a standard gaussian ! p_x = %f" % p_x
    assert qx - -0.61175641 < 10e-5, "q_x doesn't implement a mixture of gaussians ! q_x = %f" % q_x
