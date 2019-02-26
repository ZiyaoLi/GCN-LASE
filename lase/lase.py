import tensorflow as tf

from lase.models import Model
import lase.layers as layers
from lase.aggregators import Gate, SageAggregator, RandWalkAggregator, ConcatAggregator

flags = tf.app.flags
FLAGS = flags.FLAGS


class LASE(Model):
    """Implementation of LASE."""
    def __init__(self, layer_infos, features, sample_placeholders, adj_placeholders,
                 num_classes, num_nodes=None, num_links=None, init_trans_dim=None,
                 node_identity_dim=0, link_identity_dim=0,
                 dropout=0., aggregator_type="sage", sigmoid_loss=False,
                 **kwargs):

        super(LASE, self).__init__(**kwargs)

        assert aggregator_type in ("sage", "rw", "concat"), \
            "Unknown aggregator: %s" % aggregator_type
        self.aggregator_type = aggregator_type

        # get info from placeholders...
        self.sample_placeholders = sample_placeholders
        self.samples = sample_placeholders['samples']
        self.adj_placeholders = adj_placeholders

        assert num_nodes is not None or features['node'] is not None, \
            'Must specify the maximum number of nodes if missing node features.'
        assert num_links is not None or features['link'] is not None, \
            'Must specify the maximum number of links if missing link features.'
        self.num_nodes = features['node'].shape[0] if features['node'] is not None else num_nodes
        self.num_links = features['link'].shape[0] if features['link'] is not None else num_links
        self.node_feats = self._get_feats_variable(features['node'], self.num_nodes, node_identity_dim, 'node')
        self.link_feats = self._get_feats_variable(features['link'], self.num_links, link_identity_dim, 'link')
        self.node_feat_dim = self.node_feats.shape.as_list()[-1]
        self.link_feat_dim = self.link_feats.shape.as_list()[-1]

        self.num_classes = num_classes  # int
        self.sigmoid_loss = sigmoid_loss   # bool
        self.dropout = dropout  # float
        self.init_trans_dim = FLAGS.dim_0 \
            if init_trans_dim is None else init_trans_dim

        self.num_layers = len(layer_infos)
        self.layer_infos = layer_infos  # a copy of layer_infos
        self.hidden_dims = [self.init_trans_dim
                            if aggregator_type == 'rw'
                            else self.node_feat_dim]
        for layer_info in layer_infos:
            if self.aggregator_type != 'rw' and layer_info.combination_type == 'concat':
                self.hidden_dims.append(layer_info.output_dim * 2)
            else:
                self.hidden_dims.append(layer_info.output_dim)
        # self.hidden_dims: [feat_dim, hidden_dim_1, hidden_dim_2, ...]

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.gate_layers = []
        self.aggregators = []
        self.pred_layer = None
        self._build_layers()

        self.est_hidden = None
        self.est_logits = None
        self.est_pred = None
        self.loss = 0.
        self._build_expectational_model()

        self.gate_weights = []
        self.min_var_weights = []
        self.full_hidden = None
        self.full_pred = None
        self._build_full_model()

    def _get_feats_variable(self, const_feats, num_identities, identity_dim=0, prefix=None):
        embed_var_name = prefix + '_embeddings'
        if identity_dim > 0:
            self.vars[embed_var_name] = \
                tf.get_variable(embed_var_name, [num_identities, identity_dim])
        else:
            self.vars[embed_var_name] = None

        if const_feats is None:
            if identity_dim == 0:
                raise Exception("Must have a positive value for %s identity feature dimension "
                                "if no constant features given." % prefix)
            else:
                return self.vars[embed_var_name]
        else:
            features = tf.Variable(
                tf.constant(const_feats, dtype=tf.float32), trainable=False)
            if self.vars[embed_var_name] is not None:
                features = tf.concat(
                    [features, self.vars[embed_var_name]], axis=1)
            return features

    def _build_layers(self):
        for i, layer_info in enumerate(self.layer_infos):
            input_dims = [self.hidden_dims[i], self.hidden_dims[i], self.link_feat_dim]
            gate_layer = Gate(dims=input_dims, bias=False, name='gate_%d' % i)
            if self.aggregator_type == "sage":
                aggregator = SageAggregator(input_dims=input_dims,
                                            output_dim=layer_info.output_dim,
                                            gate_layer=gate_layer,
                                            combination_type=layer_info.combination_type,
                                            dropout=self.dropout, bias=True,
                                            name='sageaggregator_%d' % i)
            elif self.aggregator_type == "rw":
                aggregator = RandWalkAggregator(input_dims=[self.node_feat_dim] + input_dims,
                                                output_dim=layer_info.output_dim,
                                                gate_layer=gate_layer,
                                                dropout=self.dropout, bias=True,
                                                name='randwalkaggregator_%d' % i)
            else:
                aggregator = ConcatAggregator(input_dims=input_dims,
                                              output_dim=layer_info.output_dim,
                                              gate_layer=gate_layer,
                                              combination_type=layer_info.combination_type,
                                              dropout=self.dropout, bias=True,
                                              name='concataggregator_%d' % i)
            self.gate_layers.append(gate_layer)
            self.aggregators.append(aggregator)
        self.pred_layer = layers.Dense(input_dim=self.hidden_dims[-1],
                                       output_dim=self.num_classes,
                                       dropout=self.dropout,
                                       act=None)
        if self.aggregator_type == 'rw':
            self.init_trans_layer = layers.Dense(input_dim=self.node_feat_dim,
                                                 output_dim=self.init_trans_dim,
                                                 dropout=self.dropout,
                                                 act=None)

    def _build_expectational_model(self):
        features = [tf.nn.embedding_lookup(self.node_feats, sample['nodes']) for sample in self.samples]
        if self.aggregator_type == 'rw':
            hiddens = [self.init_trans_layer(feat) for feat in features]
        else:
            hiddens = features
        for l in range(self.num_layers):
            next_hiddens = []
            for hop in range(self.num_layers - l):
                self_feats = features[hop]
                self_vecs = hiddens[hop]
                neigh_vecs = hiddens[hop + 1]
                link_vecs = tf.nn.embedding_lookup(self.link_feats, self.samples[hop + 1]['links'])
                select_probs = self.samples[hop + 1]['probs']
                if self.aggregator_type == 'rw':
                    h = self.aggregators[l]((self_feats, self_vecs, neigh_vecs, link_vecs, select_probs))
                else:
                    h = self.aggregators[l]((self_vecs, neigh_vecs, link_vecs, select_probs))
                next_hiddens.append(h)
            hiddens = next_hiddens

        self.est_hidden = tf.nn.l2_normalize(hiddens[0], axis=-1)
        self.est_logits = self.pred_layer(self.est_hidden)
        self.est_pred = self._predict(self.est_logits)

        self._loss()
        self._optimize()

    def _build_full_model(self):
        if self.aggregator_type == 'rw':
            hidden = self.init_trans_layer(self.node_feats)
        else:
            hidden = self.node_feats

        for i in range(self.num_layers):
            self_vecs = hidden
            neigh_vecs = tf.nn.embedding_lookup(hidden, self.adj_placeholders['nodes'])
            link_vecs = tf.nn.embedding_lookup(self.link_feats, self.adj_placeholders['links'])
            if self.aggregator_type == 'rw':
                self.gate_weights.append(
                    self.aggregators[i].gate_sampling_weights((self_vecs, neigh_vecs, link_vecs)))
                self.min_var_weights.append(
                    self.aggregators[i].min_var_sampling_weights((self.node_feats, self_vecs, neigh_vecs, link_vecs)))
                hidden = self.aggregators[i]((self.node_feats, self_vecs, neigh_vecs, link_vecs, None))
            else:
                self.gate_weights.append(
                    self.aggregators[i].gate_sampling_weights((self_vecs, neigh_vecs, link_vecs)))
                self.min_var_weights.append(
                    self.aggregators[i].min_var_sampling_weights((self_vecs, neigh_vecs, link_vecs)))
                hidden = self.aggregators[i]((self_vecs, neigh_vecs, link_vecs, None))

        self.full_hidden = tf.nn.l2_normalize(hidden, axis=-1)
        full_predict_logits = self.pred_layer(self.full_hidden)
        self.full_pred = self._predict(full_predict_logits)

    def _loss(self):
        # Weight decay loss
        for i in range(self.num_layers):
            for var in self.aggregators[i].vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
            for var in self.gate_layers[i].vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.pred_layer.vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # classification loss
        if self.sigmoid_loss:
            self.loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.est_logits, labels=self.sample_placeholders['labels']))
        else:
            self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.est_logits, labels=self.sample_placeholders['labels']))

        tf.summary.scalar('loss', self.loss)

    def _optimize(self):
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                  for grad, var in grads_and_vars]
        # self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)

    def _predict(self, logits):
        if self.sigmoid_loss:
            return tf.nn.sigmoid(logits)
        else:
            return tf.nn.softmax(logits)


class FullBatchLASE(LASE):
    """Implementation of LASE, a full-batch training version."""
    def __init__(self, layer_infos, features, placeholders,
                 num_classes, num_nodes=None, num_links=None, init_trans_dim=None,
                 node_identity_dim=0, link_identity_dim=0,
                 dropout=0., aggregator_type="sage", sigmoid_loss=False,
                 **kwargs):

        super(LASE, self).__init__(**kwargs)

        assert aggregator_type in ("sage", "rw", "concat"), \
            "Unknown aggregator: %s" % aggregator_type
        self.aggregator_type = aggregator_type

        # get info from placeholders...
        self.adj_nodes = placeholders['adj_nodes']
        self.adj_links = placeholders['adj_links']
        self.batch_nodes = placeholders['nodes']
        self.batch_labels = placeholders['labels']

        assert num_nodes is not None or features['node'] is not None, \
            'Must specify the maximum number of nodes if missing node features.'
        assert num_links is not None or features['link'] is not None, \
            'Must specify the maximum number of links if missing link features.'
        self.num_nodes = features['node'].shape[0] if features['node'] is not None else num_nodes
        self.num_links = features['link'].shape[0] if features['link'] is not None else num_links
        self.node_feats = self._get_feats_variable(features['node'], self.num_nodes, node_identity_dim, 'node')
        self.link_feats = self._get_feats_variable(features['link'], self.num_links, link_identity_dim, 'link')
        self.node_feat_dim = self.node_feats.shape.as_list()[-1]
        self.link_feat_dim = self.link_feats.shape.as_list()[-1]

        self.num_classes = num_classes  # int
        self.sigmoid_loss = sigmoid_loss   # bool
        self.dropout = dropout  # float
        self.init_trans_dim = FLAGS.dim_0 \
            if init_trans_dim is None else init_trans_dim

        self.num_layers = len(layer_infos)
        self.layer_infos = layer_infos  # a copy of layer_infos
        self.hidden_dims = [self.init_trans_dim
                            if aggregator_type == 'rw'
                            else self.node_feat_dim]
        for layer_info in layer_infos:
            if self.aggregator_type != 'rw' and layer_info.combination_type == 'concat':
                # the output of current layer is concatenated
                self.hidden_dims.append(layer_info.output_dim * 2)
            else:
                self.hidden_dims.append(layer_info.output_dim)
        # self.hidden_dims: [feat_dim, hidden_dim_1, hidden_dim_2, ...]

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.gate_layers = []
        self.aggregators = []
        self.pred_layer = None
        self.init_trans_layer = None
        self._build_layers()

        self.batch_hidden = None
        self.batch_logits = None
        self.batch_pred = None
        self.loss = 0.
        self._build_model()

        self._loss()
        self._optimize()  # defined in LASE

    def _build_model(self):
        if self.aggregator_type == 'rw':
            hidden = self.init_trans_layer(self.node_feats)
        else:
            hidden = self.node_feats

        for i in range(self.num_layers):
            self_vecs = hidden
            neigh_vecs = tf.nn.embedding_lookup(hidden, self.adj_nodes)
            link_vecs = tf.nn.embedding_lookup(self.link_feats, self.adj_links)
            if self.aggregator_type == 'rw':
                hidden = self.aggregators[i]((self.node_feats, self_vecs, neigh_vecs, link_vecs, None))
            else:
                hidden = self.aggregators[i]((self_vecs, neigh_vecs, link_vecs, None))

        self.batch_hidden = tf.nn.l2_normalize(tf.nn.embedding_lookup(hidden, self.batch_nodes))
        self.batch_logits = self.pred_layer(self.batch_hidden)
        self.batch_pred = self._predict(self.batch_logits)

    def _build_expectational_model(self):
        raise NotImplementedError

    def _build_full_model(self):
        raise NotImplementedError

    def _loss(self):
        # Weight decay loss
        for i in range(self.num_layers):
            for var in self.aggregators[i].vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
            for var in self.gate_layers[i].vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.pred_layer.vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # classification loss
        if self.sigmoid_loss:
            self.loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.batch_logits, labels=self.batch_labels))
        else:
            self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.batch_logits, labels=self.batch_labels))

        tf.summary.scalar('loss', self.loss)

