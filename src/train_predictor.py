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


def joint_predictor_maker():
    W1 = tf.Variable(tf.truncated_normal(shape=(2 * N_DISCRETE, 200), stddev=0.01))
    b1 = tf.Variable(tf.zeros(200))
    W2 = tf.Variable(tf.truncated_normal(shape=(200, 200), stddev=0.01))
    b2 = tf.Variable(tf.zeros(200))
    W3 = tf.Variable(tf.truncated_normal(shape=(200, N_DISCRETE), stddev=0.01))
    b3 = tf.Variable(tf.zeros(1))

    def join_predictor(inp):
        out1 = tf.nn.relu(tf.matmul(inp, W1) + b1)
        out2 = tf.nn.relu(tf.matmul(out1, W2) + b2)
        out3 = tf.matmul(out2, W3) + b3
        return out3
    return join_predictor


def mse(out, target, axis=None):
    return tf.reduce_mean((out - target) ** 2, axis=axis)


# def discretize(ten, mini, maxi, n):
#     arg = tf.clip_by_value(tf.cast(tf.floor(n * (ten - mini) / (maxi - mini)), tf.int32), 0, n - 1)
#     return tf.squeeze(tf.one_hot(arg, n))


def bell(x):
    return tf.pow(tf.cos(x * np.pi), 20)


def discretize(ten, mini, maxi, n):
    x = tf.linspace(0.0, 1.0, n)
    arg = (ten - mini) / (maxi - mini)
    x = x - arg
    return bell(x)


dataset_t0 = database.get_dataset(path, positions=True, actions=True)
dataset_t0 = dataset_t0.prefetch(5 * batch_size)
dataset_t1 = dataset_t0.skip(1)
dataset = tf.data.Dataset.zip((dataset_t0, dataset_t1))
dataset = dataset.batch(batch_size)
dataset = dataset.shuffle(buffer_size=buffer_size)
dataset = dataset.repeat()

iterator = dataset.make_initializable_iterator()
batch_t0, batch_t1 = iterator.get_next()

positions = tf.split(batch_t0["positions"], 4, axis=1)
actions = tf.split(batch_t0["actions"], 4, axis=1)
positions_target = tf.split(batch_t1["positions"], 4, axis=1)

discrete_positions = [discretize(x, -3.16, 3.16, N_DISCRETE) for x in positions]
discrete_actions = [discretize(x, -3, 3, N_DISCRETE) for x in actions]
discrete_positions_target = [discretize(x, -3.16, 3.16, N_DISCRETE) for x in positions_target]

joint_predictors = [joint_predictor_maker() for a in actions]
inps = [tf.concat([p, a], axis=1) for p, a in zip(discrete_positions, discrete_actions)]
outs = [joint_predictor(inp) for inp, joint_predictor in zip(inps, joint_predictors)]
losses = [mse(out, target) for out, target in zip(outs, discrete_positions_target)]
ops = [tf.train.AdamOptimizer(5e-4).minimize(loss) for loss in losses]

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run([iterator.initializer, tf.global_variables_initializer()])
    for i in range(args.n_batches):
        _, l = sess.run([ops, losses])
        if i % 200 == 0:
            print(l)
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
    path = saver.save(sess, args.output + "{}_nd{}/".format(st, args.n_discrete))
    print("Network saved under {}".format(path))

with open(path + "/input_sequencs_path.txt", "w") as f:
    f.write(args.input)
