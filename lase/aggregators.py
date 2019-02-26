import tensorflow as tf

from lase.layers import Layer
from lase.inits import glorot, zeros

from lase.tf_ops import tf_vector_l2_norm, tf_vector_mul, tf_matrix_mul


class Gate(Layer):
    def __init__(self, dims, bias=False, act=tf.nn.sigmoid, name=None, **kwargs):

        super(Gate, self).__init__(**kwargs)

        self.bias = bias
        self.act = act

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['self_weights'] = glorot((dims[0],), name='self_weights')
            self.vars['neigh_weights'] = glorot((dims[1],), name='neigh_weights')
            self.vars['link_weights'] = glorot((dims[2],), name='link_weights')
            if self.bias:
                self.vars['bias'] = zeros(1, name='bias')

        if self.logging:
            self._log_vars()

        self.input_dims = dims
        self.output_dim = None

    def _call(self, inputs):
        self_vecs, neigh_vecs, link_vecs = inputs
        trans_self = tf_vector_mul(self_vecs, self.vars['self_weights'])
        trans_neigh = tf_vector_mul(neigh_vecs, self.vars['neigh_weights'])
        trans_link = tf_vector_mul(link_vecs, self.vars['link_weights'])
        if self.bias:
            gate_tmp1 = tf.expand_dims(tf.add(trans_self, self.vars['bias']), axis=-1)
        else:
            gate_tmp1 = tf.expand_dims(trans_self, axis=-1)
        gate_tmp2 = tf.add(trans_neigh, trans_link)
        gate = self.act(tf.add(gate_tmp1, gate_tmp2))
        return gate


class SageAggregator(Layer):
    """
    Aggregates with SAGE NN.
    Args:
        input_dims: (self_dim, neigh_dim, link_dim).
        output_dim: output vector dimension.
        gate_layer: a Gate object, if None, build a new, default one according to input_dims.
        combination_type: 'concat', 'add' or 'elem_mul'.
        dropout: dropout rate.
        bias: bool, whether to use bias or not.
        act: activation function.
    """

    def __init__(self, input_dims, output_dim, gate_layer=None, combination_type='concat',
                 dropout=0., bias=False, act=tf.nn.relu, name=None, **kwargs):
        super(SageAggregator, self).__init__(**kwargs)

        assert combination_type in ('concat', 'add', 'elem_mul'), \
            'Unknown combination type: ' + combination_type
        self.combination_type = combination_type
        self.dropout = dropout
        self.bias = bias
        self.act = act

        self.input_dims = input_dims
        self.output_dim = 2 * output_dim if combination_type == 'concat' else output_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if gate_layer is None:
            self.gate_layer = Gate(input_dims, name=(self.name + name + '_gate'))
        else:
            self.gate_layer = gate_layer

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['self_weights'] = glorot((input_dims[0], output_dim),
                                               name='self_weights')
            self.vars['neigh_weights'] = glorot((input_dims[1], output_dim),
                                                name='neigh_weights')
            self.vars['link_weights'] = glorot((input_dims[2], input_dims[1]),
                                               name='link_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):

        self_vecs, neigh_vecs, link_vecs, select_probs = inputs

        self_vecs = tf.nn.dropout(self_vecs, 1 - self.dropout)
        neigh_vecs = tf.nn.dropout(neigh_vecs, 1 - self.dropout)
        link_vecs = tf.nn.dropout(link_vecs, 1 - self.dropout)

        gate_values = self.gate_layer([self_vecs, neigh_vecs, link_vecs])

        trans_links = tf.nn.sigmoid(tf_matrix_mul(link_vecs, self.vars['link_weights']))
        amplified_neighs = tf.multiply(neigh_vecs, trans_links)

        if select_probs is not None:  # monte carlo expectation of sum of neighbors
            gate_div_prob = tf.divide(gate_values, select_probs)
            weighted_amp_neighs = tf.multiply(amplified_neighs, tf.expand_dims(gate_div_prob, axis=-1))
            expected_sum_neighs = tf.reduce_mean(weighted_amp_neighs, -2)
        else:  # directly calculation of sum of neighbors
            weighted_amp_neighs = tf.multiply(amplified_neighs, tf.expand_dims(gate_values, axis=-1))
            expected_sum_neighs = tf.reduce_sum(weighted_amp_neighs, -2)

        from_self = tf_matrix_mul(self_vecs, self.vars["self_weights"])
        from_neighs = tf_matrix_mul(expected_sum_neighs, self.vars['neigh_weights'])

        if self.combination_type == 'add':
            output = tf.add(from_self, from_neighs)
        elif self.combination_type == 'concat':
            output = tf.concat([from_self, from_neighs], axis=-1)
        else:
            output = tf.multiply(from_self, from_neighs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

    def gate_sampling_weights(self, inputs):
        self_vecs, neigh_vecs, link_vecs = inputs

        self_vecs = tf.nn.dropout(self_vecs, 1 - self.dropout)
        neigh_vecs = tf.nn.dropout(neigh_vecs, 1 - self.dropout)
        link_vecs = tf.nn.dropout(link_vecs, 1 - self.dropout)

        gate_values = self.gate_layer([self_vecs, neigh_vecs, link_vecs])

        return gate_values

    def min_var_sampling_weights(self, inputs):
        self_vecs, neigh_vecs, link_vecs = inputs

        self_vecs = tf.nn.dropout(self_vecs, 1 - self.dropout)
        neigh_vecs = tf.nn.dropout(neigh_vecs, 1 - self.dropout)
        link_vecs = tf.nn.dropout(link_vecs, 1 - self.dropout)

        gate_values = self.gate_layer([self_vecs, neigh_vecs, link_vecs])

        trans_links = tf.nn.sigmoid(tf_matrix_mul(link_vecs, self.vars['link_weights']))
        amplified_neighs = tf.multiply(neigh_vecs, trans_links)
        amplified_neigh_norms = tf_vector_l2_norm(amplified_neighs)

        return tf.multiply(gate_values, amplified_neigh_norms)


class ConcatAggregator(Layer):
    """
    Aggregates with SAGE NN.
    Args:
        input_dims: (self_dim, neigh_dim, link_dim).
        output_dim: output vector dimension.
        gate_layer: a Gate object, if None, build a new, default one according to input_dims.
        combination_type: 'concat', 'add' or 'elem_mul'.
        dropout: dropout rate.
        bias: bool, whether to use bias or not.
        act: activation function.
    """

    def __init__(self, input_dims, output_dim, gate_layer=None, combination_type='concat',
                 dropout=0., bias=False, act=tf.nn.relu, name=None, **kwargs):
        super(ConcatAggregator, self).__init__(**kwargs)

        assert combination_type in ('concat', 'add', 'elem-mul'), \
            'Unknown combination type: ' + combination_type
        self.combination_type = combination_type
        self.dropout = dropout
        self.bias = bias
        self.act = act

        self.input_dims = input_dims
        self.output_dim = 2 * output_dim if combination_type == 'concat' else output_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if gate_layer is None:
            self.gate_layer = Gate(input_dims, name=(self.name + name + '_gate'))
        else:
            self.gate_layer = gate_layer

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['self_weights'] = glorot((input_dims[0], output_dim),
                                               name='self_weights')
            self.vars['neigh_weights'] = glorot((input_dims[1] + input_dims[2], output_dim),
                                                name='neigh_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):

        self_vecs, neigh_vecs, link_vecs, select_probs = inputs

        self_vecs = tf.nn.dropout(self_vecs, 1 - self.dropout)
        neigh_vecs = tf.nn.dropout(neigh_vecs, 1 - self.dropout)
        link_vecs = tf.nn.dropout(link_vecs, 1 - self.dropout)

        gate_values = self.gate_layer([self_vecs, neigh_vecs, link_vecs])

        concat_neighs = tf.concat([neigh_vecs, link_vecs], axis=-1)

        if select_probs is not None:  # monte carlo expectation of sum of neighbors
            gate_div_prob = tf.divide(gate_values, select_probs)
            weighted_amp_neighs = tf.multiply(concat_neighs, tf.expand_dims(gate_div_prob, axis=-1))
            expected_sum_neighs = tf.reduce_mean(weighted_amp_neighs, -2)
        else:  # directly calculation of sum of neighbors
            weighted_amp_neighs = tf.multiply(concat_neighs, tf.expand_dims(gate_values, axis=-1))
            expected_sum_neighs = tf.reduce_sum(weighted_amp_neighs, -2)

        from_self = tf_matrix_mul(self_vecs, self.vars["self_weights"])
        from_neighs = tf_matrix_mul(expected_sum_neighs, self.vars['neigh_weights'])

        if self.combination_type == 'add':
            output = tf.add(from_self, from_neighs)
        elif self.combination_type == 'concat':
            output = tf.concat([from_self, from_neighs], axis=-1)
        else:
            output = tf.multiply(from_self, from_neighs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

    def gate_sampling_weights(self, inputs):
        self_vecs, neigh_vecs, link_vecs = inputs

        self_vecs = tf.nn.dropout(self_vecs, 1 - self.dropout)
        neigh_vecs = tf.nn.dropout(neigh_vecs, 1 - self.dropout)
        link_vecs = tf.nn.dropout(link_vecs, 1 - self.dropout)

        gate_values = self.gate_layer([self_vecs, neigh_vecs, link_vecs])

        return gate_values

    def min_var_sampling_weights(self, inputs):
        self_vecs, neigh_vecs, link_vecs = inputs

        self_vecs = tf.nn.dropout(self_vecs, 1 - self.dropout)
        neigh_vecs = tf.nn.dropout(neigh_vecs, 1 - self.dropout)
        link_vecs = tf.nn.dropout(link_vecs, 1 - self.dropout)

        gate_values = self.gate_layer([self_vecs, neigh_vecs, link_vecs])

        concat_neighs = tf.concat([neigh_vecs, link_vecs], axis=-1)
        concat_neigh_norms = tf_vector_l2_norm(concat_neighs)

        return tf.multiply(gate_values, concat_neigh_norms)


class RandWalkAggregator(Layer):
    """
        Aggregates with SAGE NN.
        Args:
            input_dims: (self_feature_dim, self_hidden_dim, neigh_hidden_dim, link_dim).
            output_dim: output vector dimension.
            gate_layer: a Gate object, if None, build a new, default one according to input_dims.
            combination_type: 'concat', 'add' or 'elem_mul'.
            dropout: dropout rate.
            bias: bool, whether to use bias or not.
            act: activation function.
        """

    def __init__(self, input_dims, output_dim, gate_layer=None,
                 dropout=0., bias=False, act=tf.nn.relu, name=None, **kwargs):
        super(RandWalkAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act

        self.input_dims = input_dims
        self.output_dim = output_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if gate_layer is None:
            self.gate_layer = Gate(input_dims, name=(self.name + name + '_gate'))
        else:
            self.gate_layer = gate_layer

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['self_weights'] = glorot((input_dims[0], input_dims[2]),
                                               name='self_weights')
            self.vars['neigh_weights'] = glorot((input_dims[2], output_dim),
                                                name='neigh_weights')
            self.vars['link_weights'] = glorot((input_dims[3], input_dims[2]),
                                               name='link_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):

        self_feats, self_vecs, neigh_vecs, link_vecs, select_probs = inputs

        self_feats = tf.nn.dropout(self_feats, 1 - self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1 - self.dropout)
        neigh_vecs = tf.nn.dropout(neigh_vecs, 1 - self.dropout)
        link_vecs = tf.nn.dropout(link_vecs, 1 - self.dropout)

        gate_values = self.gate_layer([self_vecs, neigh_vecs, link_vecs])

        trans_links = tf.nn.sigmoid(tf_matrix_mul(link_vecs, self.vars['link_weights']))
        trans_self_feats = tf_matrix_mul(self_feats, self.vars['self_weights'])
        amplified_neighs = tf.multiply(
            tf.multiply(neigh_vecs, trans_links),
            tf.expand_dims(trans_self_feats, -2)
        )

        if select_probs is not None:  # monte carlo expectation of sum of neighbors
            gate_div_prob = tf.divide(gate_values, select_probs)
            weighted_amp_neighs = tf.multiply(amplified_neighs, tf.expand_dims(gate_div_prob, axis=-1))
            sum_neighs = tf.reduce_mean(weighted_amp_neighs, -2)
        else:  # directly calculation of sum of neighbors
            weighted_amp_neighs = tf.multiply(amplified_neighs, tf.expand_dims(gate_values, axis=-1))
            sum_neighs = tf.reduce_sum(weighted_amp_neighs, -2)

        output = sum_neighs

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

    def gate_sampling_weights(self, inputs):
        self_vecs, neigh_vecs, link_vecs = inputs

        self_vecs = tf.nn.dropout(self_vecs, 1 - self.dropout)
        neigh_vecs = tf.nn.dropout(neigh_vecs, 1 - self.dropout)
        link_vecs = tf.nn.dropout(link_vecs, 1 - self.dropout)

        gate_values = self.gate_layer([self_vecs, neigh_vecs, link_vecs])

        return gate_values

    def min_var_sampling_weights(self, inputs):
        self_feats, self_vecs, neigh_vecs, link_vecs = inputs

        self_feats = tf.nn.dropout(self_feats, 1 - self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1 - self.dropout)
        neigh_vecs = tf.nn.dropout(neigh_vecs, 1 - self.dropout)
        link_vecs = tf.nn.dropout(link_vecs, 1 - self.dropout)

        gate_values = self.gate_layer([self_vecs, neigh_vecs, link_vecs])

        trans_links = tf_matrix_mul(link_vecs, self.vars['link_weights'])
        trans_self_feats = tf_matrix_mul(self_feats, self.vars['self_weights'])
        amplified_neighs = tf.multiply(
            tf.multiply(neigh_vecs, trans_links),
            tf.expand_dims(trans_self_feats, -2)
        )
        amplified_neigh_norms = tf_vector_l2_norm(amplified_neighs)

        return tf.multiply(gate_values, amplified_neigh_norms)


if __name__ == '__main__':

    batch_size = 100
    sample_1 = 10
    sample_2 = 25
    node_feat_dim = 50
    link_feat_dim = 60

    self_vecs = tf.zeros([batch_size, sample_1, node_feat_dim])
    neigh_vecs = tf.zeros([batch_size, sample_1, sample_2, node_feat_dim])
    link_vecs = tf.zeros([batch_size, sample_1, sample_2, link_feat_dim])
    probs = tf.ones([batch_size, sample_1, sample_2]) / sample_2

    dims = [node_feat_dim, node_feat_dim, link_feat_dim]

    gate_layer = Gate(dims=dims, bias=True)
    aggregator = SageAggregator(input_dims=dims, output_dim=60, gate_layer=gate_layer, bias=True)

    t = aggregator([self_vecs, neigh_vecs, link_vecs, probs])

    gates = aggregator.gate_sampling_weights([self_vecs, neigh_vecs, link_vecs])
    min_var_probs = aggregator.min_var_sampling_weights([self_vecs, neigh_vecs, link_vecs])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    t_ = sess.run([t, gates, min_var_probs])

    pass

