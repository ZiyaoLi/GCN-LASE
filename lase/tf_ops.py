import tensorflow as tf


def tf_vector_l2_norm(a):
    """Calculates a's (* x n) 'vectorized' l2 norm, outputs shape (*)"""
    return tf.sqrt(tf.reduce_sum(tf.multiply(a, a), axis=-1))


def tf_vector_mul(a, b):
    """Calculates A x b with A (* x n) and b (n), outputs shape (*) """
    a_shape = a.get_shape().as_list()
    b_shape = b.get_shape().as_list()
    assert a_shape[-1] == b_shape[0], \
        "Wrong input shapes: " + str(a_shape) + " and " + str(b_shape)
    return tf.reduce_sum(tf.multiply(a, b), axis=-1)


def tf_matrix_mul(a, b):
    """Calculates A x B with A (* x m x n) and B (n x p), outputs shape (* x m x p)"""
    a_shape = a.get_shape().as_list()
    b_shape = b.get_shape().as_list()
    n = a_shape[-1]
    n_ = b_shape[0]
    p = b_shape[1]
    assert n == n_, \
        "Wrong input shapes: " + str(a_shape) + " and " + str(b_shape)
    reshape_a = tf.reshape(a, shape=(-1, n))
    output_to_reshape = tf.matmul(reshape_a, b)
    # reset output shape
    a_shape[0] = -1 if a_shape[0] is None else a_shape[0]
    a_shape[-1] = p
    output = tf.reshape(output_to_reshape, shape=a_shape)
    return output
