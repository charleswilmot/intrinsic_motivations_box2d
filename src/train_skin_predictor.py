import tensorflow as tf
import database
import numpy as np
import parsers
import time
import os
import datetime
from networks import predictor_maker, mse


parser = parsers.train_parser
args = parser.parse_args()


path = args.input
batch_size = args.batch_size
buffer_size = batch_size * 5
tactile_map_length = np.load(args.input + "/tactile_map/length.npy")


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

tactile_map_predictor = predictor_maker(tactile_map_length, tactile_map_length)
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
    path = saver.save(sess, args.output + "tactile_t0__tactile_t1_{}/".format(st))
    print("Network saved under {}".format(path))

with open(path + "/input_sequencs_path.txt", "w") as f:
    f.write(args.input)
