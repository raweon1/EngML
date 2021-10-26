import numpy as np
from matplotlib.colors import LogNorm
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


class GMM:
    def __init__(self, k):
        self.k = k
        self.w = None
        self.mu = None
        self.sigma = None
        self.d = None

    def _init_param(self, X):
        self.w = np.random.uniform(size=self.k)
        self.w = self.w / self.w.sum()
        self.d = X.shape[1]
        # self.mu = np.random.uniform(0, 10, size=(self.k, self.d, 1))
        self.mu = X[np.random.choice(X.shape[0], size=3)][:, :, None]
        # self.sigma = np.random.randn(self.k, self.d, self.d)
        # self.sigma = np.tile(np.cov(data.T), (self.k, 1, 1))
        self.sigma = np.zeros((self.k, self.d, self.d))
        for i in range(self.k):
            self.sigma[i] = np.cov(X[np.random.choice(X.shape[0], size=X.shape[0] // self.k)].T)

    def _mvn(self, mu, sigma, d, X):
        # X = X.reshape(-1, 1)
        # mu = mu.reshape(-1, 1)
        a = 1 / ((2 * np.pi) ** (d / 2) * np.linalg.det(sigma) ** 0.5)
        b = np.exp(-0.5 * np.dot(np.dot((X - mu).T, np.linalg.inv(sigma)), X - mu))
        return a * b

    def _mvn_2(self, mu, sigma, d, X):
        mu = mu.reshape(1, -1)
        print(X)
        print(mu)
        print(X.shape, mu.shape)
        print((X - mu).shape)
        print((X - mu).T.shape)
        print(np.dot((X - mu).T, np.linalg.inv(sigma)).shape)
        a = 1 / ((2 * np.pi) ** (d / 2) * np.linalg.det(sigma) ** 0.5)
        b = np.exp(-0.5 * np.dot(np.dot((X - mu), np.linalg.inv(sigma)), (X - mu).T))
        return a * b

    def _estimation(self, X):
        gamma = np.zeros((X.shape[0], self.k))
        for k, (w, mu, sigma) in enumerate(zip(self.w, self.mu, self.sigma)):
            for i, sample in enumerate(X):
                gamma[i, k] = self._mvn(mu, sigma, self.d, sample[:, None])
        return gamma / np.sum(gamma, axis=1, keepdims=True)

    def _maximisation(self, X, gamma):
        self.w = np.sum(gamma, axis=0) / gamma.shape[0]
        self.mu = np.dot(gamma.T, X)[:, :, None] / (gamma.shape[0] * self.w)[:, None, None]
        for k in range(self.k):
            self.sigma[k] = np.zeros((self.d, self.d))
            for i, sample in enumerate(X):
                sample = sample[:, None]
                self.sigma[k] += gamma[i, k] * np.dot(sample - self.mu[k], (sample - self.mu[k]).T)
        self.sigma = self.sigma / (gamma.shape[0] * self.w)[:, None, None]

    def fit(self, X, max_iter=1000):
        self._init_param(X)
        for i in range(max_iter):
            gamma = self._estimation(X)
            self._maximisation(X, gamma)

    def score_samples(self, X):
        foo = np.sum(self._estimation(X), axis=1)
        print(foo)
        return foo


if __name__ == "__main__":
    data = load_iris()["data"][:, [2, 3]]
    gmm = GMM(3)
    gmm.fit(data, max_iter=100)
    # plt.scatter(data[:, 0], data[:, 1], c="blue")
    plt.scatter(gmm.mu[:, 0], gmm.mu[:, 1], c="red")

    x = np.linspace(-8, 15)
    y = np.linspace(-5, 7)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    print(XX.shape)
    Z = -gmm.score_samples(XX)
    print(Z.shape)
    Z = Z.reshape(X.shape)

    CS = plt.contour(
        X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10)
    )
    CB = plt.colorbar(CS, shrink=0.8, extend="both")
    plt.scatter(data[:, 0], data[:, 1], 0.8)

    plt.title("Negative log-likelihood predicted by a GMM")
    plt.axis("tight")
    plt.show()
