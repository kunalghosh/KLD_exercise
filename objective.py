import numpy as np

log = np.log

def kld(px, qx,X):
    """
    Implements the KL Divergence between p_x and q_x

    .. math ::
        D_{KL}(P(x)||Q(X)) = \sum_{x \in X} P(x) \log(P(x) / Q(x))

    px and qx should be instances of px() and qx() respectively, 
    implementing the distribution interface.
    """
    return (px.pdf(X) * (log(px.pdf(X)) - log(qx.pdf(X)))).sum()

    

