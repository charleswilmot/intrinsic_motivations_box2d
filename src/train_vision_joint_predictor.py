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

dataset = database.get_dataset(path, positions=True, vision=True)
dataset = dataset.map(database.discretize_dataset(N_DISCRETE))
dataset = dataset.map(database.vision_to_float32)
dataset = dataset.prefetch(5 * batch_size)
dataset = dataset.shuffle(buffer_size=buffer_size)
dataset = dataset.batch(batch_size)
dataset = dataset.repeat()

iterator = dataset.make_initializable_iterator()
batch = iterator.get_next()

size = np.prod(batch["vision"].get_shape().as_list()[1:])
vision = tf.reshape(batch["vision"], [-1, size])
discrete_positions = [tf.squeeze(x, axis=1) for x in tf.split(batch["positions"], 4, axis=1)]

predictors = [predictor_maker(size, N_DISCRETE) for pos in discrete_positions]
outs = [pred(vision) for pred in predictors]
losses = [mse(out, pos) for out, pos in zip(outs, discrete_positions)]
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
    path = saver.save(sess, args.output + "vision_t0__joint_t0_{}_nd{}/".format(st, args.n_discrete))
    print("Network saved under {}".format(path))

with open(path + "/input_sequencs_path.txt", "w") as f:
    f.write(args.input)
