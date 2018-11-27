import tensorflow as tf
import database
import numpy as np
import argparse
import time
import os
import re
import datetime
import viewer
from networks import predictor_maker, mse
from tools import future_return


ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')

parser = parsers.test_parser
args = parser.parse_args()


path = args.input
batch_size = args.batch_size
buffer_size = batch_size * 5
regex = r"sf[0-9]+.?[0-9]*_re[0-9]+_ae[0-9]+_n([0-9]+)_chunk[0-9]+.tfr"
n_records = int(re.match(regex, os.listdir(path + '/positions')[0]).group(1))
tactile_map_length = np.load(args.input + "/tactile_map/length.npy")


dataset_t0 = database.get_dataset(path, tactile_map=True, vision=True)
dataset_t0 = dataset_t0.prefetch(5 * batch_size)
dataset_t1 = dataset_t0.skip(1)
dataset = tf.data.Dataset.zip((dataset_t0, dataset_t1))
dataset = dataset.batch(batch_size)

iterator = dataset.make_initializable_iterator()
batch_t0, batch_t1 = iterator.get_next()

tactile_map = batch_t0["tactile_map"]
target_tactile_map = batch_t1["tactile_map"]

tactile_map_predictor = predictor_maker(tactile_map_length, tactile_map_length)
out = tactile_map_predictor(tactile_map)
loss = mse(out, target_tactile_map, axis=-1)

saver = tf.train.Saver()

loss_skin = np.zeros(n_records - 1)

with tf.Session() as sess:
    sess.run(iterator.initializer)
    saver.restore(sess, args.network_path)
    try:
        while True:
            ind, l = sess.run([batch_t0["index"], loss])
            loss_skin[ind] = l
    except tf.errors.OutOfRangeError:
        pass
    rturn = future_return(loss_skin, args.gamma)
    win = viewer.VisionSkinReturnWindow(rturn)
    win.set_return_lim([0, 0.1])
    sess.run(iterator.initializer)
    contact_last = False
    try:
        while True:
            args_batch = sess.run([batch_t0["vision"], target_tactile_map, out, batch_t0["index"]])
            for args in zip(*args_batch):
                win.update(*args)
                if np.max(args[1]) > 0 or contact_last:
                    time.sleep(1)
                contact_last = (np.max(args[1]) > 0)
    except tf.errors.OutOfRangeError:
        pass
