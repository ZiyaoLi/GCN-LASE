from __future__ import division
from __future__ import print_function

import numpy as np

np.random.seed(123)


class MinibatchIterator(object):
    
    """ 
    This minibatch iterator iterates over nodes for supervised learning.

    G -- networkx graph
    id2idx -- dict mapping node ids to integer values indexing feature tensor
    placeholders -- standard tensorflow placeholders object for feeding
    label_map -- map from node ids to class values (integer or list)
    num_classes -- number of output classes
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists

    Methods:
        _make_label_vec: change a list/int label into an ndarray
        construct_adj: return an adj-ndarray and a degree ndarray with a fixed neighbor size
        construct_test_adj: similarly
        end: boolean, whether an epoch is end
        batch_feed_dict: return a feed_dict according to given list of nodes
            feed_dict: {
                "batch_size": int, batch size
                "batch": a list of node indices
                "labels": a batch_size * num_classes ndarray
            }
        node_val_feed_dict: return a feed_dict with all
                            (or a fixed number of random)
                            val/test nodes
        incremental_node_val_feed_dict: return a previous, sequential result with
                                        fixed size, and whether it is finished
        num_training_batches: number of training batches
    """
    def __init__(self, graph, id_maps, label_map, sample_placeholders,
                 num_classes=None, batch_size=100, max_degree=25):

        self.G = graph
        self.nodes = graph.nodes()
        self.node_id_map = id_maps['node']
        self.link_id_map = id_maps['link']
        self.num_nodes = len(self.node_id_map)
        self.num_links = len(self.link_id_map)
        self.label_map = label_map
        self.sample_placeholders = sample_placeholders

        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0  # How many batches have been generated.

        if num_classes is None:
            self.num_classes = len(list(label_map.values())[0])
        else:
            self.num_classes = num_classes

        self.node_train_adj, self.link_train_adj, self.train_deg = self._construct_adj(train=True)
        self.node_full_adj, self.link_full_adj, self.full_deg = self._construct_adj(train=False)
        print("Max degree in graph: %d" % np.max(self.full_deg))

        self.train_nodes = [n for n in self.G.nodes() if self.G.node[n]['state'] == 'train']
        num_original_train_set = len(self.train_nodes)
        self.val_nodes = [n for n in self.G.nodes() if self.G.node[n]['state'] == 'val']
        self.test_nodes = [n for n in self.G.nodes() if self.G.node[n]['state'] == 'test']

        # don't train on nodes that only have edges to test set
        self.train_nodes = [n for n in self.train_nodes if self.train_deg[self.node_id_map[n]] > 0]

        # self.train_node_idxs = [self.node_id_map[n] for n in self.train_nodes]
        # self.val_node_idxs = [self.node_id_map[n] for n in self.val_nodes]
        # self.test_node_idxs = [self.node_id_map[n] for n in self.val_nodes]

        print("Data Sizes: train - %d (%d); val - %d; test - %d" %
              (len(self.train_nodes), num_original_train_set, len(self.val_nodes), len(self.test_nodes)))

    def _construct_adj(self, train):
        """
        Generates adjacent lists for train/val/test nodes.
            For nodes with 0 neighbors, manually add a dummy one.
        """
        # set default neighbors as the dummy neighbor
        node_adj = self.num_nodes * np.ones(shape=(self.num_nodes + 1, self.max_degree))
        link_adj = self.num_links * np.ones(shape=(self.num_nodes + 1, self.max_degree))
        deg = np.zeros(shape=(self.num_nodes,))

        for node_id in self.G.nodes():

            node_idx = self.node_id_map[node_id]

            if train and self.G.node[node_id]['state'] == 'train':
                # remove unavailable links in the training set
                neigh_ids = [neigh_id for neigh_id in self.G.neighbors(node_id)
                             if not self.G[neigh_id][node_id]['train_removed']]
            else:
                neigh_ids = [neigh_id for neigh_id in self.G.neighbors(node_id)]

            deg[node_idx] = len(neigh_ids)
            if len(neigh_ids) == 0:  # no neighbors
                continue
            elif len(neigh_ids) > self.max_degree:  # sufficient neighbors
                neigh_ids = np.random.choice(neigh_ids, self.max_degree, replace=False)
            elif len(neigh_ids) < self.max_degree:  # insufficient neighbors
                neigh_ids = np.random.choice(neigh_ids, self.max_degree, replace=True)

            neigh_node_idxs = [self.node_id_map[neigh_id] for neigh_id in neigh_ids]
            neigh_link_idxs = [self.link_id_map[self.G[neigh_id][node_id]['id']]
                               for neigh_id in neigh_ids]
            node_adj[node_idx, :] = neigh_node_idxs
            link_adj[node_idx, :] = neigh_link_idxs
        return node_adj, link_adj, deg

    def num_training_batches(self):
        return int(np.ceil(len(self.train_nodes) / self.batch_size))

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_nodes)

    def batch_feed_dict(self, batch_ids):
        labels = np.vstack([self.label_map[node_id] for node_id in batch_ids])
        batch_idxs = np.array([self.node_id_map[node_id] for node_id in batch_ids], dtype=np.int32)

        feed_dict = {
            self.sample_placeholders['batch_size']: len(batch_ids),
            self.sample_placeholders['labels']: labels
        }

        return feed_dict, batch_idxs, labels

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_nodes))
        batch_nodes = self.train_nodes[start_idx: end_idx]

        feed_dict, nodes, labels = self.batch_feed_dict(batch_nodes)
        return feed_dict, nodes, labels

    def node_val_feed_dict(self, size=None, test=False):
        if test:
            val_nodes = self.test_nodes
        else:
            val_nodes = self.val_nodes
        if size is not None:
            val_nodes = np.random.choice(val_nodes, size, replace=True)

        feed_dict, nodes, labels = self.batch_feed_dict(val_nodes)
        return feed_dict, nodes, labels

    def incremental_node_val_feed_dict(self, size, iter_num, test=False):
        if test:
            val_nodes = self.test_nodes
        else:
            val_nodes = self.val_nodes
        val_node_subset = val_nodes[iter_num * size:
                                    min((iter_num + 1) * size, len(val_nodes))]

        feed_dict, nodes, labels = self.batch_feed_dict(val_node_subset)
        is_last = (iter_num+1)*size >= len(val_nodes)
        return feed_dict, nodes, labels, is_last, val_node_subset

    def incremental_embed_feed_dict(self, size, iter_num):
        node_list = self.nodes
        val_nodes = node_list[iter_num * size:
                              min((iter_num + 1) * size, len(node_list))]

        feed_dict, nodes, labels = self.batch_feed_dict(val_nodes)
        is_last = (iter_num+1)*size >= len(node_list)
        return feed_dict, nodes, labels, is_last, val_nodes

    def shuffle(self):
        """
        Re-shuffle the training set. Also reset the batch number.
        """
        self.train_nodes = np.random.permutation(self.train_nodes)
        self.batch_num = 0


class FullbatchIterator(object):
    """
    This minibatch iterator iterates over nodes for supervised learning.

    G -- networkx graph
    id2idx -- dict mapping node ids to integer values indexing feature tensor
    placeholders -- standard tensorflow placeholders object for feeding
    label_map -- map from node ids to class values (integer or list)
    num_classes -- number of output classes
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists

    Methods:
        _make_label_vec: change a list/int label into an ndarray
        construct_adj: return an adj-ndarray and a degree ndarray with a fixed neighbor size
        construct_test_adj: similarly
        end: boolean, whether an epoch is end
        batch_feed_dict: return a feed_dict according to given list of nodes
            feed_dict: {
                "batch_size": int, batch size
                "batch": a list of node indices
                "labels": a batch_size * num_classes ndarray
            }
        node_val_feed_dict: return a feed_dict with all
                            (or a fixed number of random)
                            val/test nodes
        incremental_node_val_feed_dict: return a previous, sequential result with
                                        fixed size, and whether it is finished
        num_training_batches: number of training batches
    """

    def __init__(self, graph, id_maps, label_map,
                 num_classes=None, max_degree=25):

        self.G = graph
        self.nodes = graph.nodes()
        self.node_id_map = id_maps['node']
        self.link_id_map = id_maps['link']
        self.num_nodes = len(self.node_id_map)
        self.num_links = len(self.link_id_map)
        self.label_map = label_map

        self.max_degree = max_degree
        self.placeholders = None

        if num_classes is None:
            self.num_classes = len(list(label_map.values())[0])
        else:
            self.num_classes = num_classes

        self.node_train_adj, self.link_train_adj, self.train_deg = self._construct_adj(train=True)
        self.node_full_adj, self.link_full_adj, self.full_deg = self._construct_adj(train=False)

        self.train_nodes = [n for n in self.G.nodes() if self.G.node[n]['state'] == 'train']
        num_original_train_set = len(self.train_nodes)
        self.val_nodes = [n for n in self.G.nodes() if self.G.node[n]['state'] == 'val']
        self.test_nodes = [n for n in self.G.nodes() if self.G.node[n]['state'] == 'test']

        # don't train on nodes that only have edges to test set
        self.train_nodes = [n for n in self.train_nodes if self.train_deg[self.node_id_map[n]] > 0]

        # self.train_node_idxs = [self.node_id_map[n] for n in self.train_nodes]
        # self.val_node_idxs = [self.node_id_map[n] for n in self.val_nodes]
        # self.test_node_idxs = [self.node_id_map[n] for n in self.val_nodes]

        print("Data Sizes: train - %d (%d); val - %d; test - %d" %
              (len(self.train_nodes), num_original_train_set, len(self.val_nodes), len(self.test_nodes)))

    def _construct_adj(self, train):
        """
        Generates adjacent lists for train/val/test nodes.
            For nodes with 0 neighbors, manually add a dummy one.
        """
        # set default neighbors as the dummy neighbor
        node_adj = self.num_nodes * np.ones(shape=(self.num_nodes + 1, self.max_degree))
        link_adj = self.num_links * np.ones(shape=(self.num_nodes + 1, self.max_degree))
        deg = np.zeros(shape=(self.num_nodes,))

        for node_id in self.G.nodes():

            node_idx = self.node_id_map[node_id]

            if train and self.G.node[node_id]['state'] == 'train':
                # remove unavailable links in the training set
                neigh_ids = [neigh_id for neigh_id in self.G.neighbors(node_id)
                             if not self.G[neigh_id][node_id]['train_removed']]
            else:
                neigh_ids = [neigh_id for neigh_id in self.G.neighbors(node_id)]

            deg[node_idx] = len(neigh_ids)
            if len(neigh_ids) == 0:  # no neighbors
                continue
            elif len(neigh_ids) > self.max_degree:  # sufficient neighbors
                neigh_ids = np.random.choice(neigh_ids, self.max_degree, replace=False)
            elif len(neigh_ids) < self.max_degree:  # insufficient neighbors
                neigh_ids = np.random.choice(neigh_ids, self.max_degree, replace=True)

            neigh_node_idxs = [self.node_id_map[neigh_id] for neigh_id in neigh_ids]
            neigh_link_idxs = [self.link_id_map[self.G[neigh_id][node_id]['id']]
                               for neigh_id in neigh_ids]
            node_adj[node_idx, :] = neigh_node_idxs
            link_adj[node_idx, :] = neigh_link_idxs
        return node_adj, link_adj, deg

    def assign_placeholders(self, placeholders):
        self.placeholders = placeholders

    def batch_feed_dict(self, batch_ids):

        assert self.placeholders is not None, "Placeholders unassigned!"

        labels = np.vstack([self.label_map[node_id] for node_id in batch_ids])
        batch_idxs = np.array([self.node_id_map[node_id] for node_id in batch_ids], dtype=np.int32)

        feed_dict = {
            self.placeholders['batch_size']: len(batch_ids),
            self.placeholders['nodes']: batch_idxs,
            self.placeholders['labels']: labels
        }

        return feed_dict, batch_idxs, labels

    def get_feed_dict(self, state='train'):
        if state == 'train':
            batch_nodes = self.train_nodes
        elif state == 'val':
            batch_nodes = self.val_nodes
        else:
            batch_nodes = self.test_nodes

        feed_dict, nodes, labels = self.batch_feed_dict(batch_nodes)

        if state == 'train':
            feed_dict[self.placeholders['adj_nodes']] = self.node_train_adj
            feed_dict[self.placeholders['adj_links']] = self.link_train_adj
        else:
            feed_dict[self.placeholders['adj_nodes']] = self.node_full_adj
            feed_dict[self.placeholders['adj_links']] = self.link_full_adj

        return feed_dict, nodes, labels
