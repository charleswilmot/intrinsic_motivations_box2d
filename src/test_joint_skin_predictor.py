import time
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
regex = r".*[0-9]+_[0-9]+_[0-9]+_[0-9]+_[0-9]+_[0-9]+_nd([0-9]+)"
dirname = os.path.basename(os.path.normpath(args.network_path))
N_DISCRETE = int(re.match(regex, dirname).group(1))
regex = r"sf[0-9]+.?[0-9]*_re[0-9]+_ae[0-9]+_n([0-9]+)_chunk[0-9]+.tfr"
n_records = int(re.match(regex, os.listdir(path + '/positions')[0]).group(1))
tactile_map_length = np.load(args.input + "/tactile_map/length.npy")

dataset_t0 = database.get_dataset(path, positions=True, actions=True, tactile_map=True, vision=True)
dataset_t0 = dataset_t0.map(database.discretize_dataset(N_DISCRETE))
dataset_t1 = dataset_t0.skip(1)
dataset = tf.data.Dataset.zip((dataset_t0, dataset_t1))
dataset = dataset.batch(batch_size)

iterator = dataset.make_initializable_iterator()
batch_t0, batch_t1 = iterator.get_next()

size = 4 * N_DISCRETE
discrete_positions = tf.reshape(batch_t0["positions"], (-1, size))
discrete_actions = tf.reshape(batch_t0["actions"], (-1, size))
tactile_map = batch_t0["tactile_map"]
tactile_map_target = batch_t1["tactile_map"]

inp = tf.concat([discrete_positions, discrete_actions, tactile_map], axis=-1)

in_size = tactile_map_length + 4 * 2 * N_DISCRETE
out_size = tactile_map_length
predictor = predictor_maker(in_size, out_size)
out = predictor(inp)
loss = mse(out, tactile_map_target, axis=-1)

saver = tf.train.Saver()

losses = np.zeros(n_records - 1)

with tf.Session() as sess:
    sess.run(iterator.initializer)
    saver.restore(sess, args.network_path)
    try:
        while True:
            ind, l = sess.run([batch_t0["index"], loss])
            losses[ind] = l
    except tf.errors.OutOfRangeError:
        pass
    rturn = future_return(losses, args.gamma)
    win = viewer.VisionSkinReturnWindow(rturn)
    win.set_return_lim([-0.002, 0.02])
    sess.run(iterator.initializer)
    contact_last = False
    try:
        while True:
            args_batch = sess.run([batch_t0["vision"], tactile_map_target, out, batch_t0["index"]])
            for args in zip(*args_batch):
                win.update(*args)
                if np.max(args[1]) > 0 or contact_last:
                    time.sleep(1)
                contact_last = (np.max(args[1]) > 0)
    except tf.errors.OutOfRangeError:
        pass
