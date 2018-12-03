import tensorflow as tf
import database
import numpy as np
import parsers
import time
import os
import datetime
from networks import mse


parser = parsers.train_parser
args = parser.parse_args()


path = args.input
batch_size = args.batch_size
buffer_size = batch_size * 5

dataset = database.get_dataset(path, vision=True)
dataset = dataset.map(database.vision_to_float32)
dataset = dataset.prefetch(5 * batch_size)
dataset = dataset.shuffle(buffer_size=buffer_size)
dataset = dataset.batch(batch_size)
dataset = dataset.repeat()

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


out = layer4
regularizer = sparsity(layer1) + sparsity(layer2) + sparsity(layer3) + sparsity(layer4)
reconstruction_loss = mse(vision, out)
loss = reconstruction_loss + 0.0001 * regularizer
op = tf.train.AdamOptimizer(5e-4).minimize(loss)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run([iterator.initializer, tf.global_variables_initializer()])
    for i in range(args.n_batches):
        _, l = sess.run([op, [loss, reconstruction_loss, regularizer]])
        if i % 200 == 0:
            print(l)
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
    path = saver.save(sess, args.output + "vision_sparse_AE_{}/".format(st))
    print("Network saved under {}".format(path))

with open(path + "/input_sequencs_path.txt", "w") as f:
    f.write(args.input)
