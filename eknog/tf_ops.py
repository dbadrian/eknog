# Copyright 2018 David B. Adrian, Mercateo AG (http://www.mercateo.com)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=========================================================================

import numpy as np
import tensorflow as tf


def trilinear_dot_product(arg0, arg1, arg2, shape_right='bnd',
                          n_dim=None, d_dim=None,
                          arg_right=2, name=None):
    # As seen below, there are some issues with dynamic shape inference in TF,
    # thus we make a lookup table to manually set it, so einsum doesn't complain
    shape_lookup = {
        'bd': [None, d_dim],
        'bnd': [None, n_dim, d_dim],
        'nd': [n_dim, d_dim],
    }

    assert arg_right == 0 or arg_right == 1 or arg_right == 2, "arg_right is not {0,1,2}."
    assert shape_right in shape_lookup, "Right shape is not 'bn', 'bnd' or 'nd"

    shape_out = 'bn' if shape_right != 'bd' else 'b'
    prod_str = 'bd,{}->{}'.format(shape_right, shape_out)

    if arg_right == 0:
        l_part = tf.multiply(arg1, arg2)
        r_part = arg0
    elif arg_right == 1:
        l_part = tf.multiply(arg0, arg2)
        r_part = arg1
    elif arg_right == 2:
        l_part = tf.multiply(arg0, arg1)
        r_part = arg2

    # Tensorflow has some problems with dynamic shape inference from dataset
    # generate tensors etc.. Manually set shapes for now
    l_part.set_shape([None, d_dim])
    r_part.set_shape(shape_lookup[shape_right])

    scores = tf.einsum(prod_str, l_part, r_part, name=name)

    # In this special case, we need to add an additional dimension
    if shape_right == 'bd':  # (b) -> (b,1)
        scores = tf.expand_dims(scores, 1)

    return scores


def complex_trilinear_dot_product(arg0_real, arg0_imag, arg1_real, arg1_imag,
                                  arg2_real, arg2_imag, shape_right='bnd',
                                  n_dim=None, d_dim=None,
                                  arg_right=2, name=None):
    """Complex-Trilinear dot product as defined in the ComplEx Paper"""
    return trilinear_dot_product(arg0_real, arg1_real, arg2_real, shape_right,
                                 n_dim, d_dim, arg_right, name) \
           + trilinear_dot_product(arg0_real, arg1_imag, arg2_imag, shape_right,
                                   n_dim, d_dim, arg_right, name) \
           + trilinear_dot_product(arg0_imag, arg1_real, arg2_imag, shape_right,
                                   n_dim, d_dim, arg_right, name) \
           - trilinear_dot_product(arg0_imag, arg1_imag, arg2_real, shape_right,
                                   n_dim, d_dim, arg_right, name)


def complex_dropout(real, imag, dropout_rate):
    # we need to make sure that the binary mask will be same for RE/IM part
    real = tf.expand_dims(real, axis=1)
    imag = tf.expand_dims(imag, axis=1)

    ns = tf.shape(real)
    cc = tf.concat([real, imag], axis=1)
    d = tf.layers.dropout(cc, rate=dropout_rate,
                          noise_shape=ns, training=True)
    r, i = tf.split(d, num_or_size_splits=2, axis=1)
    r = tf.squeeze(r, axis=1)
    i = tf.squeeze(i, axis=1)

    return r, i  # real, imag


# def project_to_unit_tensor(x):
#     # Projection onto torus by applying a modulo operation on the range 0-1.
#     # This will only run on the CPU, and thus slower (for multi gpu?)
#     return tf.floormod(x, 1.0)

def project_to_unit_tensor(x):
    """
    This can be executed on the GPU, vs floormod requires H-to-D copies?

    Projects values from R^n to a torus represented by the n-dimension
    unit-square hyperplane. Operates on a tensor as element-wise operation.
    Projection behaves as following (or like floormod :)) :
         0.17 -> 0.17
        -0.17 -> 0.83   (eq: 1 - 0.17)
         1.17 -> 0.17
         3.82 -> 0.82
        -3.82 -> 0.18   (eq: 4 - 3.82)
    :param x: nd-Tensor
    :return: projected nd-Tensor
    """
    fx = tf.floor(x)
    cx = tf.ceil(tf.abs(x))

    px = tf.maximum(x, 0)
    px_mask = tf.cast(tf.greater(x, 0), tf.float32)
    nx = tf.minimum(x, 0)
    nx_mask = tf.cast(tf.less(x, 0), tf.float32)

    return (tf.multiply(cx, nx_mask) + nx) + (px - tf.multiply(fx, px_mask))


def score_l1_torus(x):
    """L1-based score function as defined in TorusE paper."""
    return 2 * x


def score_l2_torus(x):
    """L2-based score function as defined in TorusE paper."""
    return 4 * tf.square(x)


def score_exp_l2_torus(x):
    """Exponential L2-based score function as defined in TorusE paper."""
    return tf.square(x) / 4


# DISTANCES
def distance_l1(x, y, axis=-1):
    """L1-'distance' based on L1-norm of the difference of x and y."""
    return l1_norm(x - y, axis=axis)


def distance_l1_torus(x, y, axis=-1):
    """L1-'distance' based on L1-norm of the difference of x and y on the Torus.
    As defined in the TorusE paper."""
    px = project_to_unit_tensor(x)
    py = project_to_unit_tensor(y)
    diff = tf.subtract(px, py)
    diff = tf.abs(diff)
    diff = tf.minimum(diff, 1 - diff)

    return tf.reduce_sum(diff, axis=axis)


def pairwise_square_distance(mat):
    r = tf.reduce_sum(mat * mat, 1)

    # turn r into column vector
    r = tf.reshape(r, [-1, 1])
    return r - 2 * tf.matmul(mat, tf.transpose(mat)) + tf.transpose(r)


def distance_l2(x, y, squared=False, axis=-1):
    """L2-'distance' based on L2-norm of the difference of x and y."""
    return l2_norm(x - y, squared=squared, axis=axis)


def distance_l2_torus(x, y, axis=-1):
    """L2-'distance' based on L2-norm of the difference of x and y on the Torus.
    As defined in the TorusE paper."""
    px = project_to_unit_tensor(x)
    py = project_to_unit_tensor(y)
    diff = tf.subtract(px, py)
    diff = tf.square(diff)
    diff = tf.minimum(diff, 1 - diff)

    return tf.reduce_sum(diff, axis=axis)


def distance_exp_l2_torus(x, y, axis=-1):
    """Exponential L2-'distance' based on L2-norm of the difference of x and y
    on the Torus. As defined in the TorusE paper."""
    px = project_to_unit_tensor(x)
    py = project_to_unit_tensor(y)
    px = tf.exp(tf.complex(0.0, 2 * np.pi * px))
    py = tf.exp(tf.complex(0.0, 2 * np.pi * py))

    return tf.cast(l2_norm(px - py, squared=False, axis=axis), dtype=tf.float32)


# NORMS
def select_norm_by_string(norm_type):
    if norm_type == "L2":
        n_func = tf.nn.l2_loss
    elif norm_type == "L1":
        n_func = lambda x: l1_norm(x, axis=None)
    else:
        raise NotImplementedError

    return n_func


def l1_norm(t, axis=-1):
    """L1-norm"""
    return tf.reduce_sum(tf.abs(t), axis=axis)


def l2_norm(t, squared=False, axis=-1, keepdims=False):
    """L2-norm"""
    if squared:
        return tf.reduce_sum(tf.square(t), axis=axis, keepdims=keepdims)
    else:
        return tf.sqrt(
            tf.reduce_sum(tf.square(t), axis=axis, keepdims=keepdims))


# Normalziation
def dynamic_l2_normalization(tensor, axis, normalize):
    """Utility function to make it "easier" to apply l2_normalization based on
    an argument."""
    return tf.nn.l2_normalize(tensor, dim=axis) if normalize else tensor


def emb_lookup(var, entries, normalize_axis=None, expand_dim=None,
               dropout=None):
    """Utility function to lookup embeddings, but also normalize, expan_dims,
    and apply dropout in "one call"."""
    embs = tf.nn.embedding_lookup(var, entries)

    if normalize_axis is not None:
        embs = tf.nn.l2_normalize(embs, dim=normalize_axis)

    if expand_dim is not None:
        embs = tf.expand_dims(embs, expand_dim)

    if dropout is not None:
        embs = tf.nn.dropout(embs, dropout)

    return embs


# LOSS
def max_margin_loss(t, margin, name):
    """Maximum-marging loss, or basically hinge-loss with margin param."""
    return tf.reduce_sum(tf.nn.relu(margin + t), name=name)


# UTILS
def entry_stop_gradients(target, mask):
    # mask_h = tf.abs(mask-1)
    mask_h = tf.logical_not(mask, name="inverse_mask")

    mask = tf.cast(mask, dtype=tf.float32)
    mask_h = tf.cast(mask_h, dtype=tf.float32)

    return tf.stop_gradient(mask_h * target) + mask * target


def force_broadcast(A, B, broadcast_axis):
    """This function forces axis of B to be broadcasted to fit A, and returns
       the tile'd version of B."""
    A_shape = tf.shape(A)
    dim = tf.expand_dims(A_shape[broadcast_axis], 0)

    num_dims = tf.concat([tf.constant([1], dtype=tf.int64),
                          tf.shape(tf.shape(B), out_type=tf.int64)], axis=0)
    new_shape = tf.SparseTensor(indices=[[0, broadcast_axis]], values=dim,
                                dense_shape=num_dims)
    new_shape = tf.squeeze(
        tf.sparse_tensor_to_dense(new_shape, default_value=1))

    return tf.tile(B, new_shape)


def split_triplet(triplet_batch, axis=1):
    h, r, t = tf.split(triplet_batch, num_or_size_splits=3, axis=axis)
    return tf.squeeze(h), tf.squeeze(r), tf.squeeze(t)


def rank_array_by_first_element(scores, reverse=False):
    """This function takes a 2D-Tensor of [batch, n_scores] and returns the rank
    of the pivot (first element) in regards to the sorted array (row-wise).

    By default, in sorts in decreasing order, meaning highest score == rank 1 (best)"""

    sort_op = tf.greater if not reverse else tf.less

    return tf.reduce_sum(
        tf.cast(sort_op(tf.expand_dims(scores[:, 0], 1), scores[:, 1:]),
                tf.int32), axis=1) + 1


def rank_array_by_pivots(scores, pivots, reverse=False):
    """This function takes a 2D-Tensor of [batch, n_scores] and returns the rank
    of the pivot (first element) in regards to the sorted array (row-wise).

    By default, in sorts in decreasing order, meaning highest score == rank 1 (best)"""

    sort_op = tf.greater if not reverse else tf.less

    return tf.reduce_sum(tf.cast(sort_op(pivots, scores), tf.int32), axis=1) + 1


def negative_sampling_uniform(positive_sample_batch, k_negative_samples,
                              max_range):
    shape = tf.shape(positive_sample_batch, out_type=tf.int32)
    return tf.random_uniform([shape[0], k_negative_samples], minval=0,
                             maxval=max_range, dtype=tf.int64)


def negative_sampling_uniform_with_exclusion(ids, k_negative_samples,
                                             max_range):
    # THIS GENERATES POTENTIALLY LARGE TENSORS, so only use it for relations
    # where it matters when there are only few.

    # generate an indices vector
    shape = tf.shape(ids, out_type=tf.int32)

    false_mask = tf.fill(tf.shape(ids), False)
    indices = tf.cast(tf.concat(
        [tf.expand_dims(tf.range(0, shape[0]), 1), tf.expand_dims(ids, 1)], 1),
        dtype=tf.int64)
    logits = tf.SparseTensor(indices, false_mask,
                             dense_shape=tf.cast([shape[0], max_range],
                                                 dtype=tf.int64))
    logits = tf.sparse_tensor_to_dense(logits, default_value=True)

    samples = tf.multinomial(tf.log(tf.cast(logits, tf.float16)),
                             k_negative_samples)

    return samples
