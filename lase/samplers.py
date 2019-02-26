from __future__ import division
from __future__ import print_function

from sklearn.preprocessing import Normalizer

# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# flags = tf.app.flags
# FLAGS = flags.FLAGS


np.random.seed(123)


"""
Classes that are used to sample node neighborhoods
"""


class Sampler(object):
    def __init__(self, node_adj, link_adj):

        self.node_adj = node_adj
        self.link_adj = link_adj
        self.num_neigh = node_adj.shape[1]
        self.uniform_prob = 1 / self.num_neigh
        self.ind_to_be_sample = np.array(range(self.num_neigh))
        self.normalizer = Normalizer(norm='l1')  # normalize the sample weights to probabilities

    def __call__(self, nodes, num_samples, probs=None):

        new_shape = list(nodes.shape) + [num_samples]
        nodes = np.reshape(nodes, -1).astype(np.int32)

        node_out = []
        link_out = []
        probs_out = []

        for node in nodes:
            cur_probs = probs[node] if probs is not None else None
            sampled_ind = np.random.choice(self.ind_to_be_sample,
                                           num_samples, replace=True, p=cur_probs)
            node_out.append(self.node_adj[node][sampled_ind])
            link_out.append(self.link_adj[node][sampled_ind])
            if cur_probs is not None:
                probs_out.append(cur_probs[sampled_ind])

        probs_out = np.vstack(probs_out) if probs is not None \
            else self.uniform_prob * np.ones(new_shape)

        node_out = np.reshape(np.vstack(node_out), new_shape)
        link_out = np.reshape(np.vstack(link_out), new_shape)
        probs_out = np.reshape(np.vstack(probs_out), new_shape)

        return node_out, link_out, probs_out


class FullBatchSampler(Sampler):
    # a pseudo sampler that returns all neighbors of the given node.
    def __call__(self, nodes, num_samples, probs=None):
        new_shape = list(nodes.shape) + [num_samples]
        nodes = nodes.astype(np.int32)
        probs_out = probs[nodes] if probs is not None \
            else self.uniform_prob * np.ones(new_shape)
        return self.node_adj[nodes], self.link_adj[nodes], probs_out


if __name__ == '__main__':
    node_adj = np.array(
        [[1, 2, 3, 4],
         [2, 3, 0, 1],
         [3, 4, 2, 0],
         [1, 2, 3, 0],
         [3, 2, 4, 4]]
    )
    link_adj = np.array(
        [[1, 2, 3, 4],
         [6, 7, 1, 5],
         [10, 11, 9, 2],
         [7, 10, 12, 3],
         [13, 11, 14, 14]]
    )

    np.random.seed(12333)
    probs = np.random.uniform(size=20).reshape([5, 4])
    probs = Normalizer('l1').transform(probs)

    nodes = np.array(1)

    sp = Sampler(node_adj, link_adj)

    s = [0, 0, 0, 0, 0]
    q = []
    exp = 0

    for i in range(100000):
        if not i % 10000:
            print('.')
        tmp = sp(nodes, 3, probs)
        t = tmp[0]
        q.append(np.sum(tmp[0] / tmp[2]) / 3)
        for j in t:
            s[j] += 1

    print(np.array(s)[node_adj[nodes]] / 100000 / 2)
    print(probs[nodes])

    plt.hist(np.array(q))
    plt.show()

    print(np.mean(q))





