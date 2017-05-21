import pdb
import numpy as np
import theano as th
import theano.tensor as T
from objective import kld
import distributions

mu = th.shared(value = 15.0,  name='mu', borrow=True) # Initial value of mu
sigma = th.shared(value = 2.0,  name='sigma', borrow=True) # Initial value of sigma

learning_rate = 0.01

X = np.linspace(-10,20,2000) # NOTE !!!! remember: (next line)
# that (low,high) must include initial values of mu defined above.

qx = distributions.qx(mu=mu, sigma=sigma) # gaussian
px = distributions.px(m1=0,m2=10) # mix gaussian

loss = kld(px,qx,X)
#loss = kld(qx,px,X)
g_mu  = th.grad(loss, mu)
g_sig = th.grad(loss, sigma)

updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip([mu,sigma],[g_mu, g_sig])
        ]
f = th.function([], [loss, mu, sigma], updates=updates)

for i in range(1000):
    loss,mu_val,sigma_val = f()
    if i % 100:
        print("Epoch %0.5d , KLD = %f" % (i,loss))

print("mu = {}, sigma = {}".format(mu_val, sigma_val))
print("Plotting px {}".format(px.name))
#px.plot()
fig,plt=px.plot(get_fig=True)
plt.hold(True)
print("Plotting qx {}".format(qx.name))
qx = distributions.qx(mu=mu_val, sigma=sigma_val)
qx.plot()
