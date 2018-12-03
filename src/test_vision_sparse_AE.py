import tensorflow as tf
import database
import numpy as np
import parsers
import time
import viewer
import os
import datetime
from networks import mse


parser = parsers.test_parser
args = parser.parse_args()


path = args.input
batch_size = args.batch_size
buffer_size = batch_size * 5
vision_shape = np.load(args.input + '/vision/shape.npz')

dataset = database.get_dataset(path, vision=True)
dataset = dataset.map(database.vision_to_float32)
dataset = dataset.prefetch(5 * batch_size)
dataset = dataset.batch(batch_size)

iterator = dataset.make_initializable_iterator()
batch = iterator.get_next()

size = np.prod(batch["vision"].get_shape().as_list()[1:])
vision = tf.reshape(batch["vision"], [-1, size])

W1 = tf.Variable(tf.truncated_normal(shape=(size, 300), stddev=0.01))
B1 = tf.Variable(tf.zeros(shape=(300,)))
layer1 = tf.nn.relu(tf.matmul(vision, W1) + B1)
W2 = tf.Variable(tf.truncated_normal(shape=(300, 300), stddev=0.01))
B2 = tf.Variable(tf.zeros(shape=(300,)))
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + B2)
W3 = tf.Variable(tf.truncated_normal(shape=(300, 300), stddev=0.01))
B3 = tf.Variable(tf.zeros(shape=(300,)))
layer3 = tf.nn.relu(tf.matmul(layer2, W3) + B3)
W4 = tf.Variable(tf.truncated_normal(shape=(300, size), stddev=0.01))
B4 = tf.Variable(tf.zeros(shape=(size,)))
layer4 = tf.matmul(layer3, W4) + B4


def sparsity(ten):
    return - tf.reduce_mean(tf.sqrt(tf.reduce_sum(ten ** 2, axis=-1)) / tf.reduce_sum(ten, axis=-1))


def to_uint8(arr):
    return np.clip((arr + 1) * 127.5, 0, 255).astype(np.uint8)


out = layer4
regularizer = sparsity(layer1) + sparsity(layer2) + sparsity(layer3) + sparsity(layer4)
reconstruction_loss = mse(vision, out)
loss = reconstruction_loss + 0.0001 * regularizer
op = tf.train.AdamOptimizer(5e-4).minimize(loss)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(iterator.initializer)
    saver.restore(sess, args.network_path)
    win = viewer.DoubleVisionWindow()
    try:
        while True:
            vision_batch_inp, vision_batch_out = sess.run([vision, out])
            vision_batch_inp = to_uint8(vision_batch_inp)
            vision_batch_out = to_uint8(vision_batch_out)
            for vinp, vout in zip(vision_batch_inp, vision_batch_out):
                win.update(np.reshape(vinp, vision_shape), np.reshape(vout, vision_shape))
    except tf.errors.OutOfRangeError:
        pass
