import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np


def pdf(x, mu, sigma):
    o_shape = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1])
    o = np.zeros(x.shape[0])
    for i, v in enumerate(x):
        o[i] = _mvn(mu, sigma, x.shape[-1], v)
    o = o.reshape(o_shape)
    return o


def _mvn(mu, sigma, d, X):
    a = 1 / (np.power(2 * np.pi, d / 2) * np.power(np.linalg.det(sigma), 0.5))
    b = np.exp(-0.5 * np.dot(np.dot((X - mu).T, np.linalg.inv(sigma)), X - mu))
    return a * b


x = np.linspace(0, 5, 10, endpoint=False)
y = multivariate_normal.pdf(x, mean=2.5, cov=0.5)
fig1 = plt.figure()
ax = fig1.add_subplot(111)
ax.plot(x, y)

x, y = np.mgrid[-1:1:.01, -1:1:.01]
pos = np.dstack((x, y))

rv = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
# ax2.contourf(x, y, rv.pdf(pos))
ax2.contourf(x, y, pdf(pos, np.array([0.5, -0.2]), np.array([[2.0, 0.3], [0.3, 0.5]])))
plt.show()