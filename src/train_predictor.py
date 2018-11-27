import tensorflow as tf
import database
import numpy as np
import parsers
import time
import os
import datetime
from networks import predictor_maker, mse


parser = parsers.train_parser
parser.add_argument(
    '-d', '--n-discrete',
    type=int,
    default=60,
    help="Discretization precision."
)

args = parser.parse_args()


path = args.input
batch_size = args.batch_size
buffer_size = batch_size * 5
N_DISCRETE = args.n_discrete

dataset_t0 = database.get_dataset(path, positions=True, actions=True)
dataset_t0 = dataset_t0.map(database.discretize_dataset(N_DISCRETE))
dataset_t0 = dataset_t0.prefetch(5 * batch_size)
dataset_t1 = dataset_t0.skip(1)
dataset = tf.data.Dataset.zip((dataset_t0, dataset_t1))
dataset = dataset.batch(batch_size)
dataset = dataset.shuffle(buffer_size=buffer_size)
dataset = dataset.repeat()

iterator = dataset.make_initializable_iterator()
batch_t0, batch_t1 = iterator.get_next()

discrete_positions = [tf.squeeze(x, axis=1) for x in tf.split(batch_t0["positions"], 4, axis=1)]
discrete_actions = [tf.squeeze(x, axis=1) for x in tf.split(batch_t0["actions"], 4, axis=1)]
discrete_positions_target = [tf.squeeze(x, axis=1) for x in tf.split(batch_t1["positions"], 4, axis=1)]

joint_predictors = [predictor_maker(2 * args.n_discrete, args.n_discrete) for a in discrete_actions]
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
