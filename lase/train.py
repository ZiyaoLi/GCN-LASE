from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
import numpy as np

from lase.lase import LASE
from lase.models import LayerInfo
from lase.minibatch_iter import MinibatchIterator
from lase.samplers import Sampler, FullBatchSampler
from lase.utils import load_data, l1_normalize, calc_f1, calc_cross_entropy


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('log_device_placement', False, 'Whether to log device placement.')
# core params..
flags.DEFINE_string('model', 'LASE', 'Model names. See README for possible values.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_string("model_size", "small", "Can be 'big' or 'small' which controls the hidden dimensionality.")
flags.DEFINE_string('train_prefix', '', 'prefix identifying training data. must be specified.')

# left to default values in main experiments 
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'Weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 30, 'Pre-defined maximum node degree.')

flags.DEFINE_string('sampling_strategy', 'gate', 'Strategy used to sample nodes.')
flags.DEFINE_integer('sampling_interval', 5, 'How often to conduct sampling process.')
flags.DEFINE_string('aggregator_type', 'sage', 'Which aggregator to use.')
flags.DEFINE_string('combination_type', 'concat', 'Way to combine hidden layers from self and neighbors '
                                                  '(concat, add or elem-mul).')
flags.DEFINE_integer('samples_1', 30, 'Number of samples in layer 1.')
flags.DEFINE_integer('samples_2', 30, 'Number of samples in layer 2.')
flags.DEFINE_integer('samples_3', 0, 'Number of samples in layer 3.')
flags.DEFINE_integer('dim_0', 64, 'Dimension of h^(0)')
flags.DEFINE_integer('dim_1', 64, 'Dimension of h^(1) (2x if using concat).')
flags.DEFINE_integer('dim_2', 48, 'Dimension of h^(2) (2x if using concat).')
flags.DEFINE_integer('dim_3', 32, 'Dimension of h^(3) (2x if using concat).')

flags.DEFINE_integer('node_identity_dim', 0, 'the dimension of node identity features (to be learnt).')
flags.DEFINE_integer('link_identity_dim', 0, 'the dimension of link identity features (to be learnt).')

# flags.DEFINE_boolean('random_context', True, 'Whether to use random context or direct edges')
flags.DEFINE_integer('batch_size', 512, 'Mini-batch size.')
flags.DEFINE_boolean('sigmoid', False, 'Whether to use sigmoid loss (softmax if False).')

# logging, saving, validation settings etc.
flags.DEFINE_string('base_log_dir', '.', 'Base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 5, "How often to run a validation mini-batch.")
flags.DEFINE_integer('validate_batch_size', -1, "How many nodes per validation sample.")
flags.DEFINE_integer('gpu', 1, "which gpu to use.")
flags.DEFINE_integer('print_every', 5, "How often to print training info.")
flags.DEFINE_integer('max_total_batches', 10**10, "Maximum total number of iterations")

os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

GPU_MEM_FRACTION = 0.8


def log_dir():
    directory = FLAGS.base_log_dir + "/mini-" + FLAGS.train_prefix.split("/")[-2]
    directory += "/agg-{agg:s}_combine-{cmb:s}_" \
                 "lr-{lr:0.4f}_dr-{dropout:.3f}_eta-{eta:.3f}_" \
                 "sample-{sample:s}_intv-{intv:1d}_" \
                 "d0-{d0:03d}_d1-{d1:03d}_d2-{d2:03d}_d3-{d3:03d}_" \
                 "s1-{s1:03d}_s2-{s2:03d}_s3-{s3:03d}/".format(
                    agg=FLAGS.aggregator_type, cmb=FLAGS.combination_type,
                    lr=FLAGS.learning_rate, dropout=FLAGS.dropout, eta=FLAGS.weight_decay,
                    sample=FLAGS.sampling_strategy, intv=FLAGS.sampling_interval,
                    d0=FLAGS.dim_0, d1=FLAGS.dim_1, d2=FLAGS.dim_2, d3=FLAGS.dim_3,
                    s1=FLAGS.samples_1, s2=FLAGS.samples_2, s3=FLAGS.samples_3)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def estimation_evaluate(sess, model, minibatch_iter, size,
                        sampling_probs, sampler, test=False):
    pt = time.time()
    total_losses = []
    total_preds = []
    total_labels = []
    iter_num = 0
    finished = False
    while not finished:
        val_feed_dict, val_nodes, labels, finished, _ = \
            minibatch_iter.incremental_node_val_feed_dict(size, iter_num, test)
        val_feed_dict = sampling(model.layer_infos, val_nodes, sampler,
                                 sampling_probs, model.sample_placeholders, val_feed_dict)
        val_pred, val_loss = sess.run([model.est_pred, model.loss], feed_dict=val_feed_dict)
        total_preds.append(val_pred)
        total_losses.append(val_loss)
        total_labels.append(labels)

        iter_num += 1

    total_preds = np.vstack(total_preds)
    total_labels = np.vstack(total_labels)
    f1_mic, f1_mac = calc_f1(total_labels, total_preds, FLAGS.sigmoid)
    return np.mean(total_losses), f1_mic, f1_mac, (time.time() - pt)


def full_evaluate(model_full_preds, minibatch_iter, test=False):
    pt = time.time()
    _, nodes, labels = minibatch_iter.node_val_feed_dict(size=None, test=test)
    preds = model_full_preds[nodes]
    loss = np.mean(calc_cross_entropy(labels, preds))
    f1_mic, f1_mac = calc_f1(labels, preds, FLAGS.sigmoid)
    return loss, f1_mic, f1_mac, (time.time() - pt)


def construct_placeholders(num_classes, support_shapes):
    # Define placeholders
    placeholders = {
        'labels': tf.placeholder(tf.float32, shape=(None, num_classes), name='labels'),  # an ndarray of labels
        'batch_size': tf.placeholder(tf.int32, name='batch_size'),  # number of nodes in batch
    }
    sample_phs = []
    for i, shape in enumerate(support_shapes):
        if i == 0:  # No links or probs on the first layer
            sample_ph_i = {
                'nodes': tf.placeholder(tf.int32, shape=shape, name='node_samples_%d' % i)
            }
        else:
            sample_ph_i = {
                'nodes': tf.placeholder(tf.int32, shape=shape, name='node_samples_%d' % i),
                'links': tf.placeholder(tf.int32, shape=shape, name='link_samples_%d' % i),
                'probs': tf.placeholder(tf.float32, shape=shape, name='prob_samples_%d' % i),
            }
        sample_phs.append(sample_ph_i)
    placeholders['samples'] = sample_phs
    return placeholders


def calc_support_shapes(layer_infos):
    support_shape = [None]
    support_shapes = [tuple(support_shape)]
    for layer_info in reversed(layer_infos):
        support_shape.append(layer_info.num_samples)
        support_shapes.append(tuple(support_shape))
    return tuple(support_shapes)


def sampling(layer_infos, batch_nodes, sampler, sampling_probs,
             sample_placeholders, init_feed_dict):
    nodes_to_sample = np.array(batch_nodes)

    init_feed_dict[sample_placeholders['samples'][0]['nodes']] = nodes_to_sample

    for i, layer_info in enumerate(reversed(layer_infos)):
        nodes, links, probs = sampler(nodes_to_sample, layer_info.num_samples,
                                      sampling_probs[len(layer_infos) - i - 1])
        init_feed_dict[sample_placeholders['samples'][i + 1]['nodes']] = nodes
        init_feed_dict[sample_placeholders['samples'][i + 1]['links']] = links
        init_feed_dict[sample_placeholders['samples'][i + 1]['probs']] = probs

        nodes_to_sample = nodes

    return init_feed_dict


def calc_sample_probs(sess, model, adj_feed_dict, strategy='gate'):
    pt = time.time()
    if strategy == 'gate':
        sampling_probs = sess.run(model.gate_weights, feed_dict=adj_feed_dict)
    elif strategy == 'min_var':
        sampling_probs = sess.run(model.min_var_weights, feed_dict=adj_feed_dict)
    else:
        sampling_probs = [np.ones(shape=probs.get_shape().as_list())
                          for probs in model.gate_weights]
    sampling_probs = [l1_normalize(prob) for prob in sampling_probs]
    return sampling_probs, time.time() - pt


def train(train_data):

    # parameter legibility check
    assert FLAGS.sampling_strategy in ('uniform', 'gate', 'min_var'), \
        "Unknown sampling strategy: %s" % FLAGS.sampling_strategy
    assert FLAGS.aggregator_type in ('rw', 'sage', 'concat'), \
        "Unknown aggregator type: %s" % FLAGS.aggregator_type
    assert FLAGS.combination_type in ('concat', 'add', 'elem_mul'), \
        "Unknown combination type: %s" % FLAGS.combination_type

    print('Processes saved in directory: %s' % log_dir())

    graph = train_data['graph']
    features = train_data['feats']
    id_maps = train_data['id_maps']
    label_map = train_data['label_map']
    # label_convert = train_data['label_convert']

    num_classes = len(list(label_map.values())[0])
    val_size = FLAGS.batch_size if FLAGS.validate_batch_size == -1 else FLAGS.validate_batch_size

    # pad with a zero vector as the feature of the dummy node / link
    if features['node'] is not None:
        features['node'] = np.vstack([features['node'],
                                      np.zeros((features['node'].shape[1],))])
    if features['link'] is not None:
        features['link'] = np.vstack([features['link'],
                                      np.zeros((features['link'].shape[1],))])

    # construct layer information, support shapes and placeholders
    if FLAGS.aggregator_type == 'rw':
        # use only dim_1 because random walk aggregator must have the same dimensions through layers.
        if FLAGS.samples_3 != 0:
            sample_layer_infos = [LayerInfo("node", FLAGS.combination_type, FLAGS.samples_1, FLAGS.dim_0),
                                  LayerInfo("node", FLAGS.combination_type, FLAGS.samples_2, FLAGS.dim_0),
                                  LayerInfo("node", FLAGS.combination_type, FLAGS.samples_3, FLAGS.dim_0)]
        elif FLAGS.samples_2 != 0:
            sample_layer_infos = [LayerInfo("node", FLAGS.combination_type, FLAGS.samples_1, FLAGS.dim_0),
                                  LayerInfo("node", FLAGS.combination_type, FLAGS.samples_2, FLAGS.dim_0)]
        else:
            sample_layer_infos = [LayerInfo("node", FLAGS.combination_type, FLAGS.samples_1, FLAGS.dim_0)]
    else:
        if FLAGS.samples_3 != 0:
            sample_layer_infos = [LayerInfo("node", FLAGS.combination_type, FLAGS.samples_1, FLAGS.dim_1),
                                  LayerInfo("node", FLAGS.combination_type, FLAGS.samples_2, FLAGS.dim_2),
                                  LayerInfo("node", FLAGS.combination_type, FLAGS.samples_3, FLAGS.dim_3)]
        elif FLAGS.samples_2 != 0:
            sample_layer_infos = [LayerInfo("node", FLAGS.combination_type, FLAGS.samples_1, FLAGS.dim_1),
                                  LayerInfo("node", FLAGS.combination_type, FLAGS.samples_2, FLAGS.dim_2)]
        else:
            sample_layer_infos = [LayerInfo("node", FLAGS.combination_type, FLAGS.samples_1, FLAGS.dim_1)]

    sample_support_shapes = calc_support_shapes(sample_layer_infos)
    sample_placeholders = construct_placeholders(num_classes, sample_support_shapes)

    # Create minibatch iterator
    minibatch_iter = MinibatchIterator(graph, id_maps, label_map,
                                       sample_placeholders,
                                       num_classes=num_classes,
                                       batch_size=FLAGS.batch_size,
                                       max_degree=FLAGS.max_degree)

    adj_placeholders = {
        'nodes': tf.placeholder(tf.int32, shape=minibatch_iter.node_train_adj.shape),
        'links': tf.placeholder(tf.int32, shape=minibatch_iter.link_train_adj.shape)
    }

    train_adj_feed_dict = {
        adj_placeholders['nodes']: minibatch_iter.node_train_adj,
        adj_placeholders['links']: minibatch_iter.link_train_adj
    }
    full_adj_feed_dict = {
        adj_placeholders['nodes']: minibatch_iter.node_full_adj,
        adj_placeholders['links']: minibatch_iter.link_full_adj
    }

    # Create samplers
    train_sampler = Sampler(minibatch_iter.node_train_adj, minibatch_iter.link_train_adj)
    val_sampler = Sampler(minibatch_iter.node_full_adj, minibatch_iter.link_full_adj)
    # full_sampler = FullBatchSampler(minibatch_iter.node_train_adj, minibatch_iter.link_train_adj)
    # test_sampler = FullBatchSampler(minibatch_iter.node_full_adj, minibatch_iter.link_full_adj)

    model = LASE(sample_layer_infos, features,
                 sample_placeholders, adj_placeholders,
                 num_classes=minibatch_iter.num_classes,
                 num_nodes=minibatch_iter.num_nodes + 1,
                 num_links=minibatch_iter.num_links + 1,
                 node_identity_dim=FLAGS.node_identity_dim,
                 link_identity_dim=FLAGS.link_identity_dim,
                 dropout=FLAGS.dropout,
                 aggregator_type=FLAGS.aggregator_type,
                 sigmoid_loss=FLAGS.sigmoid)

    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    config.allow_soft_placement = True
    
    # Initialize session
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir(), sess.graph)

    # Init variables
    sess.run(tf.global_variables_initializer())
    
    # Train model
    
    total_batches = 0
    total_sampling_time = 0.0
    total_optimizing_time = 0.0
    training_info = {
        'train_f1_mics': {},
        'train_losses': {},
        'val_f1_mics': {},
        'val_losses': {}
    }

    print_sample_time = 0.
    print_opt_time = 0.
    print_val_time = 0.

    for epoch in range(FLAGS.epochs):  # iterating over epochs
        minibatch_iter.shuffle()
        print('Epoch: %04d' % epoch)

        ite = 0

        while not minibatch_iter.end():     # iterating over batches

            # Calculating sampling probabilities every k epochs
            if total_batches % FLAGS.sampling_interval == 0:
                train_sampling_probs, duration = \
                    calc_sample_probs(sess, model, train_adj_feed_dict, FLAGS.sampling_strategy)
                full_sampling_probs, _ = \
                    calc_sample_probs(sess, model, full_adj_feed_dict, FLAGS.sampling_strategy)
                print("Calculating sampling probabilities at epoch", '%04d' % epoch,
                      "time=", "{:.5f}s".format(duration))

            pt0 = time.time()

            # Sampling and constructing feed dict
            init_feed_dict, batch_nodes, labels = minibatch_iter.next_minibatch_feed_dict()
            feed_dict = sampling(sample_layer_infos, batch_nodes, train_sampler,
                                 train_sampling_probs, sample_placeholders, init_feed_dict)
            pt1 = time.time()

            # Optimization process
            outs = sess.run([merged, model.opt_op, model.loss, model.est_pred], feed_dict=feed_dict)
            train_cost = outs[2]
            pt2 = time.time()

            total_sampling_time += (pt1 - pt0)
            total_optimizing_time += (pt2 - pt1)
            print_sample_time += (pt1 - pt0)
            print_opt_time += (pt2 - pt1)

            if total_batches % FLAGS.validate_iter == 0:
                pt3 = time.time()
                # Small-Scale Expectational Validation
                val_feed_dict, val_nodes, val_labels = minibatch_iter.node_val_feed_dict(val_size, test=False)
                val_feed_dict = sampling(sample_layer_infos, val_nodes, val_sampler,
                                         full_sampling_probs, sample_placeholders, val_feed_dict)
                val_pred, val_loss = sess.run([model.est_pred, model.loss], feed_dict=val_feed_dict)
                val_f1_mic, val_f1_mac = calc_f1(val_labels, val_pred, FLAGS.sigmoid)

                print_val_time += (time.time() - pt3)
                training_info['val_losses'][total_batches] = val_loss
                training_info['val_f1_mics'][total_batches] = val_f1_mic

            # Print results
            if total_batches % FLAGS.print_every == 0:
                summary_writer.add_summary(outs[0], total_batches)
                train_f1_mic, train_f1_mac = calc_f1(labels, outs[-1], FLAGS.sigmoid)
                print("Iter:", '%04d' % ite,
                      "TRAIN:loss=", "{:.4f}".format(train_cost),
                      "f1_mic=", "{:.4f}".format(train_f1_mic),
                      "f1_mac=", "{:.4f}".format(train_f1_mac),
                      "\tVAL:loss=", "{:.4f}".format(val_loss),
                      "f1_mic=", "{:.4f}".format(val_f1_mic),
                      "f1_mac=", "{:.4f}".format(val_f1_mac),
                      "\tTIME:opt=", "{:.2f}s".format(print_opt_time),
                      "sample=", "{:.2f}s".format(print_sample_time),
                      "val=", "{:.2f}s".format(print_val_time))
                print_opt_time = 0
                print_sample_time = 0
                print_val_time = 0
                training_info['train_losses'][total_batches] = train_cost
                training_info['train_f1_mics'][total_batches] = train_f1_mic

            ite += 1
            total_batches += 1
            if total_batches > FLAGS.max_total_batches:
                break
    
    print("Optimization Finished! Conducting validation and test ..")

    full_sampling_probs, duration = calc_sample_probs(sess, model, full_adj_feed_dict, FLAGS.sampling_strategy)

    est_val_loss, est_val_f1_mic, est_val_f1_mac, est_val_time = \
        estimation_evaluate(sess, model, minibatch_iter, val_size,
                            full_sampling_probs, val_sampler, test=False)
    est_val_time += duration
    exp_test_loss, exp_test_f1_mic, exp_test_f1_mac, exp_test_time = \
        estimation_evaluate(sess, model, minibatch_iter, val_size,
                            full_sampling_probs, val_sampler, test=True)
    exp_test_time += duration

    pt = time.time()
    full_preds = sess.run(model.full_pred, feed_dict=full_adj_feed_dict)
    duration = time.time() - pt

    full_val_loss, full_val_f1_mic, full_val_f1_mac, full_val_time = \
        full_evaluate(full_preds, minibatch_iter, test=False)
    full_val_time += duration
    full_test_loss, full_test_f1_mic, full_test_f1_mac, full_test_time = \
        full_evaluate(full_preds, minibatch_iter, test=True)
    full_test_time += duration

    print("Expectational validation stats:",
          "loss=", "{:.4f}".format(est_val_loss),
          "f1_micro=", "{:.4f}".format(est_val_f1_mic),
          "f1_macro=", "{:.4f}".format(est_val_f1_mac),
          "time=", "{:.4f}s".format(est_val_time))
    with open(log_dir() + "val_stats.txt", "w") as fp:
        fp.write("Expectational - loss={:.4f} f1_micro={:.4f} f1_macro={:.4f} time={:.2f}s\n".
                 format(est_val_loss, est_val_f1_mic, est_val_f1_mac, est_val_time))
        fp.write("Full - loss={:.4f} f1_micro={:.4f} f1_macro={:.4f} time={:.2f}s\n".
                 format(full_val_loss, full_val_f1_mic, full_val_f1_mac, full_val_time))
    with open(log_dir() + "test_stats.txt", "w") as fp:
        fp.write("Expectational - loss={:.5f} f1_micro={:.5f} f1_macro={:.5f} time={:.2f}s\n".
                 format(exp_test_loss, exp_test_f1_mic, exp_test_f1_mac, exp_test_time))
        fp.write("Full - loss={:.5f} f1_micro={:.5f} f1_macro={:.5f} time={:.2f}s\n".
                 format(full_test_loss, full_test_f1_mic, full_test_f1_mac, full_test_time))
    with open(log_dir() + "training_info.json", "w") as fp:
        fp.write(str(training_info))


def main(argv=None):
    print("Loading training data..")
    train_data = load_data(FLAGS.train_prefix)
    print("Done loading training data..")
    train(train_data)


if __name__ == '__main__':
    tf.app.run()
