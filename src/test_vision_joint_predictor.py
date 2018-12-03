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
regex = r"vision_t0__joint_t0_[0-9]+_[0-9]+_[0-9]+_[0-9]+_[0-9]+_[0-9]+_nd([0-9]+)"
dirname = os.path.basename(os.path.normpath(args.network_path))
N_DISCRETE = int(re.match(regex, dirname).group(1))
regex = r"sf[0-9]+.?[0-9]*_re[0-9]+_ae[0-9]+_n([0-9]+)_chunk[0-9]+.tfr"
n_records = int(re.match(regex, os.listdir(path + '/positions')[0]).group(1))

dataset = database.get_dataset(path, positions=True, vision=True)
dataset = dataset.map(database.discretize_dataset(N_DISCRETE))
dataset = dataset.batch(batch_size)

iterator = dataset.make_initializable_iterator()
batch = iterator.get_next()

size = np.prod(batch["vision"].get_shape().as_list()[1:])
vision_uint8 = batch["vision"]
vision = database.uint8_to_float32(tf.reshape(vision_uint8, [-1, size]))
discrete_positions = [tf.squeeze(x, axis=1) for x in tf.split(batch["positions"], 4, axis=1)]

predictors = [predictor_maker(size, N_DISCRETE) for pos in discrete_positions]
outs = [pred(vision) for pred in predictors]
losses = [mse(out, pos, axis=-1) for out, pos in zip(outs, discrete_positions)]

saver = tf.train.Saver()

losses_per_joint = np.zeros((n_records, 4))


with tf.Session() as sess:
    sess.run(iterator.initializer)
    saver.restore(sess, args.network_path)
    try:
        while True:
            ind, l = sess.run([batch["index"], losses])
            for j, ll in enumerate(l):
                losses_per_joint[ind, j] = ll
    except tf.errors.OutOfRangeError:
        pass
    rturn = future_return(losses_per_joint, args.gamma)
    win = viewer.VisionDJointReturnWindow(rturn[:, 0:])
    sess.run(iterator.initializer)
    try:
        while True:
            vision_batch, dp, dpt, index_batch = sess.run(
                [vision_uint8, outs, discrete_positions, batch["index"]])
            dp = np.swapaxes(np.array(dp), 0, 1)
            dpt = np.swapaxes(np.array(dpt), 0, 1)
            for v, p, pt, i in zip(vision_batch, dp, dpt, index_batch):
                win.update(v, p, pt, p, i)
    except tf.errors.OutOfRangeError:
        pass
