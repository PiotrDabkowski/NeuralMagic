import tensorflow as tf


def random_mask_gen(min_side, max_side, mask_shape):
    assert min_side > 0
    assert max_side < mask_shape[0] and max_side < mask_shape[1]
    mask_shape = map(float, mask_shape)
    sides = tf.round(tf.random_uniform((2,), float(min_side), float(max_side)))
    x_side = sides[1]
    y_side = sides[0]
    y_offset = tf.round(tf.random_uniform((), 0., mask_shape[0] - tf.cast(y_side, tf.float32)))
    x_offset = tf.round(tf.random_uniform((), 0., mask_shape[1] - tf.cast(x_side, tf.float32)))
    return matrix_select(tf.ones(map(int, mask_shape)), y_offset, y_offset+y_side, x_offset, x_offset+x_side)

def matrix_select(matrix, y_from, y_to, x_from, x_to, yx_dims=(0,1)):
    sh = matrix.get_shape().as_list()
    y_size = sh[yx_dims[0]]
    x_size = sh[yx_dims[1]]
    shape = len(sh)*[1]
    shape[yx_dims[0]] = y_size
    shape[yx_dims[1]] = x_size
    shape_x = len(sh)*[1]
    shape_x[yx_dims[1]] = x_size
    shape_y = len(sh)*[1]
    shape_y[yx_dims[0]] = y_size
    ys = tf.ones(shape) * tf.reshape(tf.range(y_size, dtype=tf.float32), shape_y)
    xs = tf.ones(shape) * tf.reshape(tf.range(x_size, dtype=tf.float32), shape_x)
    cond1 = y_from <= ys
    cond2 = ys < y_to
    cond3 = x_from <= xs
    cond4 = xs < x_to
    return matrix * tf.cast(tf.logical_and(tf.logical_and(tf.logical_and(cond1, cond2), cond3), cond4), tf.float32)


def random_matrices_gen(num, min_side, max_side, mask_shape):
    return tf.stack(tuple(random_mask_gen(min_side, max_side, mask_shape) for _ in xrange(num)))
