from matplotlib.colors import LogNorm
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt


data = load_iris()["data"][:, [0, 1]]
gmm = GaussianMixture(3, covariance_type="full", init_params="random")
gmm.fit(data)

print(gmm.covariances_.sum())

# display predicted scores by the model as a contour plot
x = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]))
y = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]))
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -gmm.score_samples(XX)
Z = Z.reshape(X.shape)

CS = plt.contour(
    X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10)
)
CB = plt.colorbar(CS, shrink=0.8, extend="both")
plt.scatter(data[:, 0], data[:, 1], 0.8)
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1])
plt.title("Negative log-likelihood predicted by a GMM")
plt.axis("tight")
plt.show()
