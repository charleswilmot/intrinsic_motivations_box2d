import tensorflow as tf
import database
import numpy as np
import argparse
import os
import re
import viewer
import parsers
from tools import future_return
from networks import predictor_maker, mse


parser = parsers.test_parser
args = parser.parse_args()

path = args.input
batch_size = args.batch_size
buffer_size = batch_size * 5
regex = r"joint_t0__joint_t1_[0-9]+_[0-9]+_[0-9]+_[0-9]+_[0-9]+_[0-9]+_nd([0-9]+)"
dirname = os.path.basename(os.path.normpath(args.network_path))
N_DISCRETE = int(re.match(regex, dirname).group(1))
regex = r"sf[0-9]+.?[0-9]*_re[0-9]+_ae[0-9]+_n([0-9]+)_chunk[0-9]+.tfr"
n_records = int(re.match(regex, os.listdir(path + '/positions')[0]).group(1))

dataset_t0 = database.get_dataset(path, positions=True, actions=True, vision=True)
dataset_t0 = dataset_t0.map(database.discretize_dataset(N_DISCRETE))
dataset_t1 = dataset_t0.skip(1)
dataset = tf.data.Dataset.zip((dataset_t0, dataset_t1))
dataset = dataset.batch(batch_size)

iterator = dataset.make_initializable_iterator()
batch_t0, batch_t1 = iterator.get_next()

discrete_positions = [tf.squeeze(x, axis=1) for x in tf.split(batch_t0["positions"], 4, axis=1)]
discrete_actions = [tf.squeeze(x, axis=1) for x in tf.split(batch_t0["actions"], 4, axis=1)]
discrete_positions_target = [tf.squeeze(x, axis=1) for x in tf.split(batch_t1["positions"], 4, axis=1)]

joint_predictors = [predictor_maker(2 * N_DISCRETE, N_DISCRETE) for a in discrete_actions]
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
    except tf.errors.OutOfRangeError:
        pass
