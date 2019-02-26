from __future__ import print_function

import numpy as np
import json
import os

from networkx.readwrite import json_graph

from sklearn import metrics
from sklearn.preprocessing import Normalizer

_normalize = {'node': True, 'link': False}


def l1_normalize(w):
    bad_rows = (np.sum(w, axis=-1) < 1e-6)
    n_bad_rows = np.sum(bad_rows)
    if n_bad_rows > 0:
        print("Warning: %d bad rows." % n_bad_rows)
    w[bad_rows] = 1
    col_sum_w = np.expand_dims(np.sum(w, -1), -1)
    w /= col_sum_w
    return w


def calc_f1(y_true, y_pred, sigmoid):
    if not sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")


def calc_cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred), axis=-1)


def make_label_vec(label, label_id2idx):
    if isinstance(label, list):
        return np.array(label)
    else:
        label_vec = np.zeros(len(label_id2idx))
        label_vec[label_id2idx[label]] = 1
        return label_vec


def construct_label_ind(label_map):
    example_label = list(label_map.values())[0]
    if isinstance(example_label, list):  # list-like labels [0, 0, 1, ..
        num_classes = len(example_label)
        label_id2idx = label_idx2id = None
    else:  # string/id-like labels (one label per node)
        num_classes = len(set(label_map.values()))
        label_id2idx = {id: idx for idx, id in enumerate(set(label_map.values()))}
        label_idx2id = [id for id in set(label_map.values())]
    new_label_map = {}
    for k, v in label_map.items():
        new_label_map[k] = make_label_vec(v, label_id2idx)
    return new_label_map, {'id2idx': label_id2idx, 'idx2id': label_idx2id}


def load_data(prefix, normalize='default'):
    if normalize == 'default':
        normalize = _normalize
    else:
        assert isinstance(normalize, dict), \
            'Wrong normalize type: must be a dict or string "default".'

    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)

    if os.path.exists(prefix + "-node-feats.npy"):
        node_feats = np.load(prefix + "-node-feats.npy")
    else:
        print("No node features present.. Only identity features will be used.")
        node_feats = None

    if os.path.exists(prefix + "-link-feats.npy"):
        link_feats = np.load(prefix + "-link-feats.npy")
    else:
        print("No link features present.. Only identity features will be used.")
        link_feats = None

    # setting up string conversions:
    #   to avoid possible confusions between int and string in source files.
    node_id_map = json.load(open(prefix + "-node-id-map.json"))
    if isinstance(G_data['nodes'][0]['id'], int):  # numerical node ids
        node_map_key_convert = lambda n: int(n)
    else:
        node_map_key_convert = lambda n: n
    node_id_map = {node_map_key_convert(k): int(v) for k, v in node_id_map.items()}

    link_id_map = json.load(open(prefix + "-link-id-map.json"))
    if isinstance(G_data['links'][0]['id'], int):  # numerical link ids
        link_map_key_convert = lambda n: int(n)
    else:
        link_map_key_convert = lambda n: n
    link_id_map = {link_map_key_convert(k): int(v) for k, v in link_id_map.items()}

    label_map = json.load(open(prefix + "-label-map.json"))
    if isinstance(list(label_map.values())[0], list):
        lab_convert = lambda n: n
    else:
        lab_convert = lambda n: int(n)
    label_map = {node_map_key_convert(k): lab_convert(v) for k, v in label_map.items()}
    label_map, label_convert = construct_label_ind(label_map)

    # Make sure the graph has edge train_removed annotations #
    print("Loaded data.. now preprocessing..")

    for edge in G.edges():
        if G.node[edge[0]]['state'] != 'train' or G.node[edge[1]]['state'] != 'train':
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    if normalize.get('node', _normalize['node']) and node_feats is not None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([node_id_map[n] for n in G.nodes()
                              if G.node[n]['state'] == 'train'])
        train_node_feats = node_feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_node_feats)
        node_feats = scaler.transform(node_feats)

    if normalize.get('link', _normalize['link']) and link_feats is not None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([link_id_map[G[edge[0]][edge[1]]['id']]
                              for edge in G.edges()
                              if not G[edge[0]][edge[1]]['train_removed']])
        train_link_feats = link_feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_link_feats)
        link_feats = scaler.transform(link_feats)

    train_data = {
        'graph': G,
        'feats': {'node': node_feats, 'link': link_feats},
        'id_maps': {'node': node_id_map, 'link': link_id_map},
        'label_map': label_map,
        'label_convert': label_convert
    }

    return train_data


if __name__ == '__main__':
    rst = load_data('../fmobile/fmobile')
    pass
