import numpy as np
from matplotlib.colors import LogNorm
from scipy.stats import multivariate_normal
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


class GMM:
    def __init__(self, k, tol=1e-5):
        self.k = k
        self.w = None
        self.mu = None
        self.sigma = None
        self.d = None
        self.tol = tol

    def _init_param(self, X):
        self.w = np.random.uniform(size=self.k)
        self.w = self.w / self.w.sum()
        # self.w = np.full(self.k, 1 / self.k)
        self.d = X.shape[1]
        self.mu = X[np.random.choice(X.shape[0], size=self.k)][:, :, None]
        self.sigma = np.zeros((self.k, self.d, self.d))
        for i in range(self.k):
            self.sigma[i] = np.cov(X[np.random.choice(X.shape[0], size=X.shape[0] // self.k)].T)

    def _mvn(self, mu, sigma, d, X):
        a = 1 / (np.power(2 * np.pi, d / 2) * np.power(np.linalg.det(sigma), 0.5))
        b = np.exp(-0.5 * np.dot(np.dot((X - mu).T, np.linalg.inv(sigma)), X - mu))
        return a * b

    def _likelihood(self, X):
        likelihood = np.zeros((X.shape[0], self.k))
        for k, (w, mu, sigma) in enumerate(zip(self.w, self.mu, self.sigma)):
            for i, sample in enumerate(X):
                likelihood[i, k] = w * self._mvn(mu, sigma, self.d, sample[:, None])
        return likelihood

    def log_likelihood_sample(self, X):
        return np.log(np.sum(self._likelihood(X), axis=1))

    def log_likelihood(self, X):
        return np.sum(self.log_likelihood_sample(X))

    def _e_step(self, likelihood):
        return likelihood / np.sum(likelihood, axis=1, keepdims=True)

    def _m_step(self, X, gamma):
        n_w = np.sum(gamma, axis=0)
        self.w = n_w / X.shape[0]
        # for k in range(self.k):
        #    self.mu[k] = (np.sum(gamma[:, k, None] * X, axis=0) / n_w[k])[:, None]
        self.mu = np.dot(gamma.T, X) / n_w[:, None]
        self.mu = self.mu[:, :, None]
        for k in range(self.k):
            self.sigma[k] = np.zeros((self.d, self.d))
            for i, sample in enumerate(X):
                sample = sample[:, None] - self.mu[k]
                self.sigma[k] += gamma[i, k] * np.dot(sample, sample.T)
        self.sigma = self.sigma / n_w[:, None, None]

    def fit(self, X, max_iter=1000):
        self._init_param(X)
        for i in range(max_iter):
            likelihood = self._likelihood(X)
            log_likelihood = np.sum(np.log(np.sum(likelihood, axis=1)))
            gamma = self._e_step(likelihood)
            self._m_step(X, gamma)
            if np.abs(log_likelihood - self.log_likelihood(X)) <= self.tol:
                break


def plot_contours(data, means, covs, title):
    """visualize the gaussian components over the data"""
    plt.figure()
    plt.plot(data[:, 0], data[:, 1], 'ko')

    delta = 0.025
    k = means.shape[0]
    x = np.arange(np.min(data[:, 0]), np.max(data[:, 0]), delta)
    y = np.arange(np.min(data[:, 1]), np.max(data[:, 1]), delta)
    x_grid, y_grid = np.meshgrid(x, y)
    coordinates = np.array([x_grid.ravel(), y_grid.ravel()]).T

    col = ['green', 'red', 'indigo', "yellow", "brown"]
    for i in range(k):
        mean = means[i]
        cov = covs[i]
        z_grid = multivariate_normal(mean, cov).pdf(coordinates).reshape(x_grid.shape)
        plt.contour(x_grid, y_grid, z_grid, colors=col[i])

    plt.title(title)
    plt.tight_layout()


def plot_combined_contours(data, gmm_score_function, means, ):
    # display predicted scores by the model as a contour plot
    x = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]))
    y = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]))
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -gmm_score_function(XX)
    Z = Z.reshape(X.shape)

    CS = plt.contour(
        X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10)
    )
    CB = plt.colorbar(CS, shrink=0.8, extend="both")
    plt.scatter(data[:, 0], data[:, 1], 0.8)
    plt.scatter(means[:, 0], means[:, 1])
    plt.title("Negative log-likelihood predicted by a GMM")
    plt.axis("tight")
    plt.tight_layout()


if __name__ == "__main__":
    data = load_iris()["data"][:, [2, 3]]
    gmm = GMM(3)
    gmm.fit(data, max_iter=10000)
    # plot_contours(data, gmm.mu[:, :, 0], gmm.sigma[:, :, 0], "hello")
    plot_combined_contours(data, gmm.log_likelihood_sample, gmm.mu)
    plt.show()
