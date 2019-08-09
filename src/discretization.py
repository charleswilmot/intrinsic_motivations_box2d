import tensorflow as tf
import numpy as np


LIMIT = 0.9


def tf_discretization_reward(tactile_goal, tactile_input):
   return tf.cast(tf.reduce_sum(tf.abs(tactile_goal - tactile_input), axis=-1) < LIMIT, tf.float32)


def np_discretization_reward(tactile_goal, tactile_input):
    return np.float32(np.sum(np.abs(tactile_goal - tactile_input), axis=-1) < LIMIT)
