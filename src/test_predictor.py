import tensorflow as tf
import database
import numpy as np
import argparse
import time
import os
import re
import datetime
import viewer


ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input',
    type=str,
    action='store', default="../data/input_sequences/data_for_dev",
    help="Input sequence."
)

parser.add_argument(
    '-n', '--network-path',
    type=str,
    action='store', default="../data/networks/2018_11_15_14_39_47_nd60/",
    help="Path to network."
)

parser.add_argument(
    '-b', '--batch-size',
    type=int,
    action='store', default=512,
    help="Maximum size of a batch."
)

parser.add_argument(
    '-o', '--output',
    type=str, action='store', default="../data/networks/{}/".format(st),
    help="Output directory. Must not exist."
)

parser.add_argument(
    '-g', '--gamma',
    type=float,
    action='store', default=0.95,
    help="gamma."
)

args = parser.parse_args()


path = args.input
batch_size = args.batch_size
buffer_size = batch_size * 5
regex = r"[0-9]+_[0-9]+_[0-9]+_[0-9]+_[0-9]+_[0-9]+_nd([0-9]+)"
dirname = os.path.basename(os.path.normpath(args.network_path))
N_DISCRETE = int(re.match(regex, dirname).group(1))
regex = r"sf[0-9]+.?[0-9]*_re[0-9]+_ae[0-9]+_n([0-9]+)_chunk[0-9]+.tfr"
n_records = int(re.match(regex, os.listdir(path + '/positions')[0]).group(1))


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


def future_return(rewards, gamma):
    ret = np.zeros_like(rewards)
    prev = 0
    for i, r in reversed(list(enumerate(rewards))):
        ret[i] = gamma * prev + r
        prev = ret[i]
    return ret


dataset_t0 = database.get_dataset(path, positions=True, actions=True, vision=True)
dataset_t1 = dataset_t0.skip(1)
dataset = tf.data.Dataset.zip((dataset_t0, dataset_t1))
dataset = dataset.batch(batch_size)

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
losses = [mse(out, target, axis=-1) for out, target in zip(outs, discrete_positions_target)]

saver = tf.train.Saver()

losses_per_joint = np.zeros((n_records - 1, 4))

with tf.Session() as sess:
    sess.run(iterator.initializer)
    saver.restore(sess, args.network_path)
    try:
        while True:
            ind, l = sess.run([batch_t0["index"], losses])
            for j, ll in enumerate(l):
                losses_per_joint[ind, j] = ll
    except tf.errors.OutOfRangeError:
        pass
    rturn = future_return(losses_per_joint, args.gamma)
    win = viewer.VisionDJointReturnWindow(rturn[:, 0:])
    sess.run(iterator.initializer)
    try:
        while True:
            vision_batch, dp, dpt, dpp, index_batch = sess.run(
                [batch_t0["vision"], outs, discrete_positions_target, discrete_positions, batch_t0["index"]])
            dp = np.swapaxes(np.array(dp), 0, 1)
            dpt = np.swapaxes(np.array(dpt), 0, 1)
            dpp = np.swapaxes(np.array(dpp), 0, 1)
            for args in zip(vision_batch, dp, dpt, dpp, index_batch):
                win.update(*args)
                # time.sleep(0.05)
            # vision_batch, positions_batch, speeds_batch, index_batch = sess.run(
            #     [batch_t0["vision"], batch_t0["positions"], batch_t0["speeds"], batch_t0["index"]])
            # for args in zip(vision_batch, positions_batch, speeds_batch, index_batch):
            #     win.update(*args)
            #     time.sleep(0.05)
    except tf.errors.OutOfRangeError:
        pass
