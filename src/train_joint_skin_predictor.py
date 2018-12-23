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
tactile_map_length = np.load(args.input + "/tactile_map/length.npy")

dataset_t0 = database.get_dataset(path, positions=True, actions=True, tactile_map=True)
dataset_t0 = dataset_t0.map(database.discretize_dataset(N_DISCRETE))
dataset_t1 = dataset_t0.skip(1)
dataset = tf.data.Dataset.zip((dataset_t0, dataset_t1))
dataset = dataset.repeat()
dataset = dataset.batch(batch_size)
dataset = dataset.shuffle(buffer_size=100)
dataset = dataset.prefetch(10)

iterator = dataset.make_initializable_iterator()
batch_t0, batch_t1 = iterator.get_next()

# size = tf.reduce_prod(tf.shape(batch_t0["positions"])[1:])
size = 4 * N_DISCRETE
discrete_positions = tf.reshape(batch_t0["positions"], (-1, size))
discrete_actions = tf.reshape(batch_t0["actions"], (-1, size))
tactile_map = batch_t0["tactile_map"]
tactile_map_target = batch_t1["tactile_map"]

inp = tf.concat([discrete_positions, discrete_actions, tactile_map], axis=-1)
# in_size = tf.shape(inp)[-1]
# out_size = tf.shape(tactile_map_target)[-1]
in_size = tactile_map_length + 4 * 2 * N_DISCRETE
out_size = tactile_map_length
predictor = predictor_maker(in_size, out_size)
out = predictor(inp)
loss = mse(out, tactile_map_target)
op = tf.train.AdamOptimizer(5e-4).minimize(loss)

saver = tf.train.Saver()

mean_loss = 0
display_every = 200

with tf.Session() as sess:
    sess.run([iterator.initializer, tf.global_variables_initializer()])
    for i in range(args.n_batches):
        _, l = sess.run([op, loss])
        mean_loss += l
        if i % display_every == 0 and i != 0:
            print("batch {}: {}".format(i, mean_loss / display_every))
            mean_loss = 0
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
    path = saver.save(sess, args.output + "joint_t0_tactile_t0__tactile_t1_{}_nd{}/".format(st, args.n_discrete))
    print("Network saved under {}".format(path))

with open(path + "/input_sequencs_path.txt", "w") as f:
    f.write(args.input)
