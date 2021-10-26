import numpy as np
import matplotlib.pyplot as plt


def load_data(path):
    return np.loadtxt(open(path, "rb"), delimiter=";", skiprows=1)


class KMeans:
    """K-Means clustering.

    Parameters
    ----------
    k : int, default=3
        The number of clusters to form.

    tol : float, default=0.001
        Tolerance with regards to Euclidean norm of the difference in the cluster centers
        of two consecutive iterations to declare convergence.

    max_iterations : int, default=300
        Maximum number of iterations of the k-means algorithm for a single run.

    seed: int, default=42
        Set the seed for the random generator.

    """
    def __init__(self, k=3, tol=0.001, max_iterations=300, seed=42):
        self._rng = np.random.default_rng(seed)
        self.k = k
        self.tolerance = tol
        self.max_iterations = max_iterations
        # array of centroid locations
        self.centroids = None

    def get_centroids(self, X, labels):
        """Returns the centroids for each cluster based on the data set X and their cluster designation labels
        X is of shape (N, M), where N is the number of instances and M the number of features.
        labels is of shape (N,), where N is the number of instances.
            labels contains the cluster label for each data point, i.e. 0 <= labels[i] < k
        Returns a numpy array of shape (k, M), where k is the number of clusters and M the number of features"""
        return np.array([np.mean(X[labels == i], axis=0) for i in range(self.k)])

    def cluster_labels(self, X, centroids):
        """Returns the cluster designation for each data point in X based on centroids
        X is of shape (N, M), where N is the number of instances and M the number of features.
        centroids is of shape (k, M), where k is the number of clusters and M the number of features
        Returns a numpy array of shape (N,), where N is the number of instances,
            which contains the cluster label for each data point, i.e. 0 <= return[i] < k"""
        return np.argmin(
            np.linalg.norm(np.array([X - centroids[i] for i in range(0, self.k)]), axis=2),
            axis=0
        )

    def fit(self, X):
        # init random labels for each data point
        labels = self._rng.integers(self.k, size=X.shape[0])
        # iteration 0 for k-Means to initialize self.centroids
        self.centroids = self.get_centroids(X, labels)
        labels = self.cluster_labels(X, self.centroids)

        for i in range(self.max_iterations):
            # calculate new cluster centroids
            centroids = self.get_centroids(X, labels)
            # get new labels for each data point based on new centroid location
            labels = self.cluster_labels(X, centroids)
            # get the distance each cluster moved
            cluster_shifts = np.linalg.norm(centroids - self.centroids, axis=1)
            # if the sum of cluster movement is less than tolerance, declare convergence and terminate k-Kmeans
            if np.sum(cluster_shifts) < self.tolerance:
                break
            self.centroids = centroids
        return self

    def predict(self, X):
        """Predicts the cluster label for each data point in X
        X is of shape (N, M), where N is the number of instances and M the number of features.
        Returns a numpy array of shape (N,), where N is the number of instances,
            which contains the cluster label for each data point, i.e. 0 <= return[i] < k"""
        return self.cluster_labels(X, self.centroids)


if __name__ == "__main__":
    X = load_data("data/data.csv")
    model = KMeans()
    model.fit(X)
    plt.scatter(X[:, 0], X[:, 1], c=model.predict(X))
    plt.scatter(model.centroids[:, 0], model.centroids[:, 1], c="red", s=100)
    plt.axis("square")
    plt.show()
