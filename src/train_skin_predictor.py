import tensorflow as tf
import database
import numpy as np
import argparse
import time
import os
import datetime


parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input',
    type=str,
    action='store', default="../data/input_sequences/data_for_dev",
    help="Input sequence."
)

parser.add_argument(
    '-b', '--batch-size',
    type=int,
    action='store', default=512,
    help="Maximum size of a batch."
)

parser.add_argument(
    '-o', '--output',
    type=str, action='store', default="../data/networks/",
    help="Output directory. Must not exist."
)

parser.add_argument(
    '-d', '--n-discrete',
    type=int,
    default=60,
    help="Discretization precision."
)

parser.add_argument(
    '-n', '--n-batches',
    type=int,
    action='store', default=5000,
    help="number of batches to train."
)

args = parser.parse_args()


path = args.input
batch_size = args.batch_size
buffer_size = batch_size * 5
N_DISCRETE = args.n_discrete
tactile_map_length = np.load(args.input + "/tactile_map/length.npy")


def tactile_map_predictor_maker():
    W1 = tf.Variable(tf.truncated_normal(shape=(tactile_map_length, 200), stddev=0.01))
    b1 = tf.Variable(tf.zeros(200))
    W2 = tf.Variable(tf.truncated_normal(shape=(200, 200), stddev=0.01))
    b2 = tf.Variable(tf.zeros(200))
    W3 = tf.Variable(tf.truncated_normal(shape=(200, tactile_map_length), stddev=0.01))
    b3 = tf.Variable(tf.zeros(1))

    def tactile_map_predictor(inp):
        out1 = tf.nn.relu(tf.matmul(inp, W1) + b1)
        out2 = tf.nn.relu(tf.matmul(out1, W2) + b2)
        out3 = tf.matmul(out2, W3) + b3
        return out3
    return tactile_map_predictor


def mse(out, target, axis=None):
    return tf.reduce_mean((out - target) ** 2, axis=axis)


dataset_t0 = database.get_dataset(path, tactile_map=True)
dataset_t0 = dataset_t0.prefetch(5 * batch_size)
dataset_t1 = dataset_t0.skip(1)
dataset = tf.data.Dataset.zip((dataset_t0, dataset_t1))
dataset = dataset.batch(batch_size)
dataset = dataset.shuffle(buffer_size=buffer_size)
dataset = dataset.repeat()

iterator = dataset.make_initializable_iterator()
batch_t0, batch_t1 = iterator.get_next()

tactile_map = batch_t0["tactile_map"]
target_tactile_map = batch_t1["tactile_map"]

tactile_map_predictor = tactile_map_predictor_maker()
out = tactile_map_predictor(tactile_map)
loss = mse(out, target_tactile_map)
op = tf.train.AdamOptimizer(5e-4).minimize(loss)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run([iterator.initializer, tf.global_variables_initializer()])
    for i in range(args.n_batches):
        _, l = sess.run([op, loss])
        if i % 200 == 0:
            print(l)
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
    path = saver.save(sess, args.output + "{}_skin/".format(st, args.n_discrete))
    print("Network saved under {}".format(path))

with open(path + "/input_sequencs_path.txt", "w") as f:
    f.write(args.input)
