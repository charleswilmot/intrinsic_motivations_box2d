import tensorflow as tf


def predictor_maker(in_size, out_size):
    W1 = tf.Variable(tf.truncated_normal(shape=(in_size, 200), stddev=0.01))
    b1 = tf.Variable(tf.zeros(200))
    W2 = tf.Variable(tf.truncated_normal(shape=(200, 200), stddev=0.01))
    b2 = tf.Variable(tf.zeros(200))
    W3 = tf.Variable(tf.truncated_normal(shape=(200, out_size), stddev=0.01))
    b3 = tf.Variable(tf.zeros(out_size))

    def tactile_map_predictor(inp):
        out1 = tf.nn.relu(tf.matmul(inp, W1) + b1)
        out2 = tf.nn.relu(tf.matmul(out1, W2) + b2)
        out3 = tf.matmul(out2, W3) + b3
        return out3
    return tactile_map_predictor


def mse(out, target, axis=None):
    return tf.reduce_mean((out - target) ** 2, axis=axis)
