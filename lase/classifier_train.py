import tensorflow as tf

from lase.utils import load_data, calc_f1
from lase.minibatch_iter import MinibatchIterator
from lase.inits import glorot
from lase.models import Model

import os
import numpy as np
import time

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
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_string('train_prefix', '', 'prefix identifying training data. must be specified.')

# left to default values in main experiments
flags.DEFINE_integer('epochs', 10, 'Number of epochs to train.')
flags.DEFINE_float('weight_decay', 0.001, 'Weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('batch_size', 256, 'Mini-batch size.')
flags.DEFINE_boolean('sigmoid', False, 'Whether to use sigmoid loss (softmax if False).')

flags.DEFINE_integer('max_degree', 1, 'Number of epochs to train.')

flags.DEFINE_string('base_log_dir', './experiments/naive', 'Base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 5, "How often to run a validation mini-batch.")
flags.DEFINE_integer('validate_batch_size', -1, "How many nodes per validation sample.")
flags.DEFINE_integer('gpu', 1, "which gpu to use.")
flags.DEFINE_integer('print_every', 5, "How often to print training info.")
flags.DEFINE_integer('max_total_batches', 10**10, "Maximum total number of iterations")


class NaiveClassifier:
    def __init__(self, placeholders, input_dim, output_dim):
        self.x = placeholders['embeddings']
        self.y = placeholders['labels']
        self.w = glorot((input_dim, output_dim), name='weight')

        self.logits = tf.matmul(self.x, self.w)

        if FLAGS.sigmoid:
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits, labels=self.y))
            self.pred = tf.nn.sigmoid(self.logits)
        else:
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.y))
            self.pred = tf.nn.softmax(self.logits)

        self.loss += FLAGS.weight_decay * tf.nn.l2_loss(self.w)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                  for grad, var in grads_and_vars]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)


def log_dir():
    directory = FLAGS.base_log_dir + "/naive-" + FLAGS.train_prefix.split("/")[-2]
    directory += "/lr{lr:0.4f}_eta{eta:.3f}_epc{epc:03d}_batch{bs:04d}/".format(
        lr=FLAGS.learning_rate,
        eta=FLAGS.weight_decay,
        epc=FLAGS.epochs,
        bs=FLAGS.batch_size
    )
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def incre_evaluate(sess, model, xs, minibatch_iter, placeholders,
                   size, test=False):
    pt = time.time()
    total_losses = []
    total_preds = []
    total_labels = []
    iter_num = 0
    finished = False
    while not finished:
        val_feed_dict, val_nodes, labels, finished, _ = \
            minibatch_iter.incremental_node_val_feed_dict(size, iter_num, test)
        val_feed_dict[placeholders['embeddings']] = xs[val_nodes]
        val_pred, val_loss = sess.run([model.pred, model.loss], feed_dict=val_feed_dict)
        total_preds.append(val_pred)
        total_losses.append(val_loss)
        total_labels.append(labels)

        iter_num += 1

    total_preds = np.vstack(total_preds)
    total_labels = np.vstack(total_labels)
    f1_mic, f1_mac = calc_f1(total_labels, total_preds, FLAGS.sigmoid)
    return np.mean(total_losses), f1_mic, f1_mac, (time.time() - pt)


def train(train_data):
    print('Processes saved in directory: %s' % log_dir())

    G = train_data['graph']
    xs = train_data['feats']['node']
    id_maps = train_data['id_maps']
    label_map = train_data['label_map']

    num_classes = len(list(label_map.values())[0])
    val_size = FLAGS.batch_size if FLAGS.validate_batch_size == -1 else FLAGS.validate_batch_size

    placeholders = {
        'labels': tf.placeholder(tf.float32, shape=(None, num_classes), name='labels'),  # an ndarray of labels
        'embeddings': tf.placeholder(tf.float32, shape=(None, xs.shape[-1]), name='embeddings'),
        'batch_size': tf.placeholder(tf.int32, name='batch_size'),  # number of nodes in batch
    }

    minibatch_iter = MinibatchIterator(G, id_maps, label_map, placeholders,
                                       num_classes=num_classes,
                                       batch_size=FLAGS.batch_size,
                                       max_degree=FLAGS.max_degree)

    model = NaiveClassifier(placeholders, xs.shape[-1], num_classes)

    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    # Initialize session
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir(), sess.graph)

    # Init variables
    sess.run(tf.global_variables_initializer())

    # Train model
    total_batches = 0
    training_info = {
        'train_f1_mics': {},
        'train_losses': {},
        'val_f1_mics': {},
        'val_losses': {}
    }

    print_time = 0.
    print_val_time = 0.

    for epoch in range(FLAGS.epochs):  # iterating over epochs
        minibatch_iter.shuffle()
        print('Epoch: %04d' % epoch)

        ite = 0

        while not minibatch_iter.end():  # iterating over batches

            pt0 = time.time()
            # Sampling and constructing feed dict
            feed_dict, batch_nodes, labels = minibatch_iter.next_minibatch_feed_dict()
            feed_dict[placeholders['embeddings']] = xs[batch_nodes]

            # Optimization process
            outs = sess.run([model.opt_op, model.loss, model.pred], feed_dict=feed_dict)
            train_cost = outs[1]
            pt1 = time.time()

            print_time += (pt1 - pt0)

            if total_batches % FLAGS.validate_iter == 0:
                pt2 = time.time()
                # Small-Scale Expectational Validation
                val_feed_dict, val_nodes, val_labels = minibatch_iter.node_val_feed_dict(val_size, test=False)
                val_feed_dict[placeholders['embeddings']] = xs[val_nodes]
                val_pred, val_loss = sess.run([model.pred, model.loss], feed_dict=val_feed_dict)
                val_f1_mic, val_f1_mac = calc_f1(val_labels, val_pred, FLAGS.sigmoid)

                print_val_time += (time.time() - pt2)
                training_info['val_losses'][total_batches] = val_loss
                training_info['val_f1_mics'][total_batches] = val_f1_mic

            # Print results
            if total_batches % FLAGS.print_every == 0:
                train_f1_mic, train_f1_mac = calc_f1(labels, outs[-1], FLAGS.sigmoid)
                print("Iter:", '%04d' % ite,
                      "TRAIN:loss=", "{:.4f}".format(train_cost),
                      "f1_mic=", "{:.4f}".format(train_f1_mic),
                      "f1_mac=", "{:.4f}".format(train_f1_mac),
                      "\tVAL:loss=", "{:.4f}".format(val_loss),
                      "f1_mic=", "{:.4f}".format(val_f1_mic),
                      "f1_mac=", "{:.4f}".format(val_f1_mac),
                      "\tTIME:opt=", "{:.2f}s".format(print_time),
                      "val=", "{:.2f}s".format(print_val_time))
                print_time = 0
                print_val_time = 0
                training_info['train_losses'][total_batches] = train_cost
                training_info['train_f1_mics'][total_batches] = train_f1_mic

            ite += 1
            total_batches += 1
            if total_batches > FLAGS.max_total_batches:
                break

    print("Optimization Finished! Conducting validation and test ..")

    val_loss, val_f1_mic, val_f1_mac, val_time = \
        incre_evaluate(sess, model, xs, minibatch_iter, placeholders, val_size, False)
    test_loss, test_f1_mic, test_f1_mac, test_time = \
        incre_evaluate(sess, model, xs, minibatch_iter, placeholders, val_size, True)

    print("Validation stats:",
          "loss=", "{:.4f}".format(val_loss),
          "f1_micro=", "{:.4f}".format(val_f1_mic),
          "f1_macro=", "{:.4f}".format(val_f1_mac),
          "time=", "{:.4f}s".format(val_time))
    print("Test stats:",
          "loss=", "{:.4f}".format(test_loss),
          "f1_micro=", "{:.4f}".format(test_f1_mic),
          "f1_macro=", "{:.4f}".format(test_f1_mac),
          "time=", "{:.4f}s".format(test_time))
    with open(log_dir() + "stats.txt", "w") as fp:
        fp.write("Validation stats:" +
                 "loss=" + "{:.4f}".format(val_loss) +
                 "f1_micro=" + "{:.4f}".format(val_f1_mic) +
                 "f1_macro=" + "{:.4f}".format(val_f1_mac) +
                 "time=" + "{:.4f}s".format(val_time))
        fp.write("Test stats:" +
                 "loss=" + "{:.4f}".format(test_loss) +
                 "f1_micro=" + "{:.4f}".format(test_f1_mic) +
                 "f1_macro=" + "{:.4f}".format(test_f1_mac) +
                 "time=" + "{:.4f}s".format(test_time))
    with open(log_dir() + "training_info.json", "w") as fp:
        fp.write(str(training_info))


def main(argv=None):
    print("Loading training data..")
    train_data = load_data(FLAGS.train_prefix)
    print("Done loading training data..")
    train(train_data)


if __name__ == '__main__':
    tf.app.run()
