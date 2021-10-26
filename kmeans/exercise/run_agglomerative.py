import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram


def load_data(path):
    return np.loadtxt(open(path, "rb"), delimiter=";", skiprows=1)


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    # source https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


class AgglomerativeClustering:
    """Agglomerative clustering.

        Parameters
        ----------
        linkage: string, default="single"
            describes which linkage mode to use to fuse clusters
            must be one of ["single", "complete", "average"]

        """
    def __init__(self, linkage="single"):
        self.linkage = linkage
        self.linkage_func = {"single": self.single_linkage,
                             "complete": self.complete_linkage,
                             "average": self.average_linkage}
        # contains the composite nodes as tuples (left_child, right_child)
        # works the same as children in
        #   https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
        self.children_ = []
        # contains the distance of between (left_child, right_child) of composite nodes
        self.distances_ = []
        self.labels_ = []

    def single_linkage(self, cluster_1, cluster_2):
        """Returns the single linkage distance of two clusters
        cluster_1 is of shape (R, M) where R is the number of data points in this cluster and M the number of features
        cluster_2 is of shape (S, M) where S is the number of data points in this cluster and M the number of features
        Returns the distance between these two clusters as float"""
        # TODO
        return 0

    def complete_linkage(self, cluster_1, cluster_2):
        """Returns the complete linkage distance of two clusters
        cluster_1 is of shape (R, M) where R is the number of data points in this cluster and M the number of features
        cluster_2 is of shape (S, M) where S is the number of data points in this cluster and M the number of features
        Returns the distance between these two clusters as float"""
        # TODO
        return 0

    def average_linkage(self, cluster_1, cluster_2):
        """Returns the average linkage distance of two clusters
        cluster_1 is of shape (R, M) where R is the number of data points in this cluster and M the number of features
        cluster_2 is of shape (S, M) where S is the number of data points in this cluster and M the number of features
        Returns the distance between these two clusters as float"""
       # TODO
        return 0

    def fit(self, X):
        # list of current clusters. each cluster in clusters is a list of data points which are in this cluster
        clusters = [[i] for i in range(X.shape[0])]
        # dictionary which assigns each cluster a unique id.
        cluster_id = {repr(cluster): i for i, cluster in enumerate(clusters)}
        # counter beginning from n_samples. n_samples + i is the id of the i-th composite cluster
        counter = itertools.count(X.shape[0])
        # cache to speed up the algorithm
        cache = {}
        while len(clusters) > 1:
            score, cluster_1, cluster_2 = np.inf, 0, 0
            # calculate the distance - based on linkage mode - for each unique cluster pair
            for i in range(len(clusters) - 1):
                for j in range(i + 1, len(clusters)):
                    key = repr(clusters[i]) + repr(clusters[j])
                    try:
                        new_score = cache[key]
                    except KeyError:
                        new_score = self.linkage_func[self.linkage](X[clusters[i]], X[clusters[j]])
                        cache[key] = new_score
                    if new_score < score:
                        score = new_score
                        cluster_1 = i
                        cluster_2 = j
            # create a new cluster which contains all data points from both clusters
            clusters.append(clusters[cluster_1] + clusters[cluster_2])
            # create a new composite cluster node as a tuple of (cluster left, cluster right)
            self.children_.append((cluster_id[repr(clusters[cluster_1])], cluster_id[repr(clusters[cluster_2])]))
            # save the distance of the new composite cluster node as dist(cluster_left, cluster_right)
            self.distances_.append(score)
            # create an id for the new composite cluster node to be used in self.children_
            cluster_id[repr(clusters[-1])] = next(counter)
            # delete merged clusters
            del clusters[cluster_2]
            del clusters[cluster_1]
        self.labels_ = np.arange(X.shape[0])
        self.children_ = np.asarray(self.children_)
        self.distances_ = np.asarray(self.distances_)


if __name__ == "__main__":
    X = load_data("../data/data.csv")
    model = AgglomerativeClustering(linkage="single")
    model.fit(X[:10])
    plot_dendrogram(model)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()

