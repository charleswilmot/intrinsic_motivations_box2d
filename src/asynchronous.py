import atexit
import environment
import socket
import multiprocessing
import subprocess
import numpy as np
from numpy import pi, log
from numpy.random import normal
from tempfile import TemporaryDirectory
import tensorflow as tf
import tensorflow.contrib.layers as tl
import time
import viewer
import os
from tensorboard import main as tb
import filelock


def actions_dict_from_array(actions):
    return {
        "Arm1_to_Arm2_Left": actions[0],
        "Ground_to_Arm1_Left": actions[1],
        "Arm1_to_Arm2_Right": actions[2],
        "Ground_to_Arm1_Right": actions[3]
    }


def get_cluster(n_parameter_servers, n_workers):
    spec = {}
    port = get_available_port(2222)
    for i in range(n_parameter_servers):
        if "ps" not in spec:
            spec["ps"] = []
        spec["ps"].append("localhost:{}".format(i + port))
    for i in range(n_workers):
        if "worker" not in spec:
            spec["worker"] = []
        spec["worker"].append("localhost:{}".format(i + port + n_parameter_servers))
    return tf.train.ClusterSpec(spec)


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def get_available_port(start_port=6006):
    port = start_port
    while is_port_in_use(port):
        port += 1
    return port


class Worker:
    def __init__(self, task_index, pipe, env, cluster, logdir, discount_factor, sequence_length, reward_params):
        self.task_index = task_index
        self.cluster = cluster
        self._n_workers = self.cluster.num_tasks("worker") - 1
        self.job_name = "worker"
        self.server = tf.train.Server(cluster, self.job_name, task_index)
        self.name = "/job:{}/task:{}".format(self.job_name, task_index)
        self.device = tf.train.replica_device_setter(worker_device=self.name, cluster=cluster)
        self.env = env
        self.discount_factor = discount_factor
        self.entropy_coef = 0.0  # 0.01
        self.sequence_length = sequence_length
        self.reward_params = reward_params
        self.pipe = pipe
        self._prev_summary_prefix_rl = None
        self._prev_summary_prefix_model = None
        self.define_networks()
        self.logdir = logdir
        # graph = tf.get_default_graph() if task_index == 0 else None
        graph = None
        self.summary_writer = tf.summary.FileWriter(self.logdir + "/worker{}".format(task_index), graph=graph)
        self.saver = tf.train.Saver()
        self.sess = tf.Session(target=self.server.target)
        if task_index == 0 and len(self.sess.run(tf.report_uninitialized_variables())) > 0:  # todo: can be done in Experiment
            self.sess.run(tf.global_variables_initializer())
            print("{}  variables initialized".format(self.name))

    def get_model_state(self):
        # returns a state from the environment
        raise NotImplementedError("This method must be overwritten.")

    def get_rl_state(self):
        # returns a state from the environment
        raise NotImplementedError("This method must be overwritten.")

    def to_rl_feed_dict(self, **kwargs):
        # transforms the inputs into a feed dict for the actor
        raise NotImplementedError("This method must be overwritten.")

    def to_model_feed_dict(self, **kwargs):
        # transforms the inputs into a feed dict for the actor
        raise NotImplementedError("This method must be overwritten.")

    def define_net_dims(self):
        # must define:
        #   - self.model_net_dim
        #   - self.rl_shared_net_dim
        #   - self.actor_remaining_net_dim
        #   - self.critic_remaining_net_dim
        raise NotImplementedError("This method must be overwritten.")

    def define_reward(self, **params):
        raise NotImplementedError("This method must be overwritten.")

    def define_networks(self):
        raise NotImplementedError("This method must be overwritten.")

    def wait_for_variables_initialization(self):
        while len(self.sess.run(tf.report_uninitialized_variables())) > 0:
            print("{}  waiting for variable initialization...".format(self.name))
            time.sleep(1)

    def __call__(self):
        cmd = self.pipe.recv()
        while not cmd == "done":
            print("{} got command {}".format(self.name, cmd))
            self.__getattribute__(cmd[0])(*cmd[1:])
            cmd = self.pipe.recv()

    def save(self, path):
        save_path = self.saver.save(self.sess, path + "/network.ckpt")
        self.pipe.send("{} saved model to {}".format(self.name, save_path))

    def restore(self, path):
        self.saver.restore(self.sess, os.path.normpath(path + "/network.ckpt"))
        self.pipe.send("{} variables restored from {}".format(self.name, path))

    def run_reinforcement_learning(self, n_updates, summary_prefix, train_actor=True):
        self.def_rl_summaries(summary_prefix)
        self.def_model_summaries(summary_prefix)
        global_rl_step = self.sess.run(self.global_rl_step)
        n_updates += global_rl_step
        while global_rl_step < n_updates - self._n_workers:
            # Collect some experience
            transitions = self.run_n_rl_steps()
            # Update the global networks
            global_rl_step = self.update_reinforcement_learning(*transitions, train_actor=train_actor)
        self.summary_writer.flush()
        self.pipe.send("{} going IDLE".format(self.name))

    def run_all(self, n_updates, summary_prefix, train_actor=True):
        self.def_rl_summaries(summary_prefix)
        self.def_model_summaries(summary_prefix)
        global_both_step = self.sess.run(self.global_both_step)
        n_updates += global_both_step
        while global_both_step < n_updates - self._n_workers:
            # Collect some experience
            transitions = self.run_n_rl_steps()
            # Update the global networks
            global_both_step = self.update_all(*transitions, train_actor=train_actor)
        self.summary_writer.flush()
        self.pipe.send("{} going IDLE".format(self.name))

    def run_display(self, sample=True):
        win = viewer.JointAgentWindow(self.discount_factor, return_lookback=50)
        while not self.pipe.poll():
            self.run_n_display_steps(win, sample)
        win.close()
        self.pipe.recv()  # done
        self.pipe.send("{} (display) going IDLE".format(self.name))

    def run_model(self, n_updates, summary_prefix):
        self.def_rl_summaries(summary_prefix)
        self.def_model_summaries(summary_prefix)
        global_model_step = self.sess.run(self.global_model_step)
        n_updates += global_model_step
        while global_model_step < n_updates - self._n_workers:
            # Collect some experience
            states = self.run_n_model_steps()
            # Update the global networks
            global_model_step = self.update_model(states)
        self.summary_writer.flush()
        self.pipe.send("{} going IDLE".format(self.name))

    def get_action(self):
        state = self.get_rl_state()
        feed_dict = self.to_rl_feed_dict(states=[state])
        action = self.sess.run(self.sample_action, feed_dict=feed_dict)
        return action[0]

    def run_n_rl_steps(self):
        model_states = []
        states = []
        actions = []
        for _ in range(self.sequence_length):
            # get states
            model_states.append(self.get_model_state())
            states.append(self.get_rl_state())
            # get action
            action = self.get_action()
            actions.append(action)
            # set action
            action_dict = actions_dict_from_array(action)
            self.env.set_positions(action_dict)
            # run environment step
            self.env.env_step()
        model_states.append(self.get_model_state())
        return model_states, states, actions

    def run_n_model_steps(self):
        states = []
        for _ in range(self.sequence_length):
            # get state
            states.append(self.get_model_state())
            # get action
            action = self.get_action()
            # run action in env
            action_dict = actions_dict_from_array(action)
            self.env.set_positions(action_dict)
            self.env.env_step()
        return states

    # TODO put that function in the subclass
    def run_n_display_steps(self, win, sample=True):
        action_return_fetches = [self.sample_action if sample else self.mu, self.critic_value]
        predicted_positions_reward_fetches = [self.model_outputs, self.rewards]
        for _ in range(self.sequence_length):
            # get current vision
            vision = self.env.vision
            # get current positions
            current_positions = self.env.discrete_positions
            # get action and predicted return
            states = [self.get_rl_state()]
            model_states = [self.get_model_state()]
            rl_feed_dict = self.to_rl_feed_dict(states=states)
            actions, predicted_returns = self.sess.run(action_return_fetches, feed_dict=rl_feed_dict)
            action = actions[0]
            predicted_return = predicted_returns[0]
            # run action in env
            action_dict = actions_dict_from_array(action)
            self.env.set_positions(action_dict)
            self.env.env_step()
            # get current positions
            next_positions = self.env.discrete_positions
            # get predicted positions and reward
            model_states.append(self.get_model_state())
            model_feed_dict = self.to_model_feed_dict(states=model_states, targets=True)
            predicted_positions, current_rewards = self.sess.run(predicted_positions_reward_fetches, feed_dict=model_feed_dict)
            current_reward = current_rewards[0]
            predicted_positions = predicted_positions[0].reshape((4, -1))
            # display
            win(vision, current_positions, predicted_positions, next_positions, current_reward, predicted_return)

    def rewards_to_return(self, rewards, prev_return=0):
        returns = np.zeros_like(rewards)
        for i in range(len(rewards) - 1, -1, -1):
            r = rewards[i]
            prev_return = r + self.discount_factor * prev_return
            returns[i] = prev_return
        return returns

    def update_reinforcement_learning(self, model_states, rl_states, actions, train_actor=True):
        feed_dict = self.to_model_feed_dict(states=model_states, targets=True)
        rewards, model_losses = self.sess.run([self.rewards, self.model_losses], feed_dict=feed_dict)
        feed_dict = self.to_rl_feed_dict(states=rl_states, actions=actions, rewards=rewards, model_losses=model_losses)
        train_op = self.rl_train_op if train_actor else self.critic_train_op
        fetches = [self.actor_loss, self.critic_loss, self.global_rl_step, train_op, self.rl_summary_stagewise]
        aloss, closs, global_rl_step, _, rl_summary = self.sess.run(fetches, feed_dict=feed_dict)
        self.summary_writer.add_summary(rl_summary, global_step=global_rl_step)
        if global_rl_step % 100 <= self._n_workers:
            self.summary_writer.flush()
        print("{} finished update number {} (actor loss = {:.3f}  critic loss = {:.3f})".format(
              self.name, global_rl_step, aloss, closs))
        return global_rl_step

    def update_model(self, states):
        feed_dict = self.to_model_feed_dict(states=states, targets=True)
        fetches = [self.model_loss, self.global_model_step, self.model_train_op, self.model_summary]
        loss, global_model_step, _, model_summary = self.sess.run(fetches, feed_dict=feed_dict)
        self.summary_writer.add_summary(model_summary, global_step=global_model_step)
        if global_model_step % 100 <= self._n_workers:
            self.summary_writer.flush()
        print("{} finished update number {} (loss = {:.3f})".format(self.name, global_model_step, loss))
        return global_model_step

    def update_all(self, model_states, rl_states, actions, train_actor=True):
        feed_dict = self.to_model_feed_dict(states=model_states, targets=True)
        fetches = [self.rewards, self.model_losses, self.model_loss, self.model_train_op, self.model_summary, self.global_both_step_inc]
        rewards, model_losses, mloss, _, model_summary, global_both_step = self.sess.run(fetches, feed_dict=feed_dict)
        self.summary_writer.add_summary(model_summary, global_step=global_both_step)
        feed_dict = self.to_rl_feed_dict(states=rl_states, actions=actions, rewards=rewards, model_losses=model_losses)
        train_op = self.rl_train_op if train_actor else self.critic_train_op
        fetches = [self.actor_loss, self.critic_loss, self.global_both_step_inc, train_op, self.rl_summary]
        aloss, closs, global_both_step, _, rl_summary = self.sess.run(fetches, feed_dict=feed_dict)
        self.summary_writer.add_summary(rl_summary, global_step=global_both_step)
        if global_both_step % 100 <= self._n_workers:
            self.summary_writer.flush()
        print("{} finished update number {} (model loss = {:.3f}  critic loss = {:.3f})".format(
              self.name, global_both_step, mloss, closs))
        return global_both_step


class BaseJointAgentWorker(Worker):
    def define_networks(self):
        self.define_net_dims()
        with tf.device(self.device):
            # define model
            self.define_model()
            # define reinforcement learning
            self.define_reinforcement_learning()

    def define_reinforcement_learning(self):
        net_dim = self.rl_shared_net_dim + self.actor_remaining_net_dim
        self.rl_inputs = tf.placeholder(shape=[None, net_dim[0]], dtype=tf.float32, name="actor_inputs")
        self.return_targets_not_bootstraped = tf.placeholder(shape=[None], dtype=tf.float32, name="returns_target")
        batch_size = tf.shape(self.rl_inputs)[0]
        constant_gammas = tf.fill(dims=[batch_size], value=self.discount_factor)
        increasing_discounted_gammas = tf.cumprod(constant_gammas, reverse=True)
        prev_layer = self.rl_inputs
        with tf.variable_scope("shared"):
            for i, d in enumerate(self.rl_shared_net_dim[1:]):
                prev_layer = tl.fully_connected(prev_layer, d, scope="layer{}".format(i))
        fork_layer = prev_layer
        with tf.variable_scope("private_to_critic"):
            for i, d in enumerate(self.critic_remaining_net_dim):
                activation_fn = tf.nn.relu if i < len(self.critic_remaining_net_dim) - 1 else None
                prev_layer = tl.fully_connected(prev_layer, d, scope="layer{}".format(i), activation_fn=activation_fn)
            self.critic_value = tf.squeeze(prev_layer, axis=1, name="critic_value")
        self.return_targets = self.return_targets_not_bootstraped + increasing_discounted_gammas * self.critic_value[-1]
        self.critic_losses = (self.critic_value - self.return_targets) ** 2
        self.critic_loss = tf.reduce_mean(self.critic_losses, name="loss")
        self.actions = tf.placeholder(shape=[None, net_dim[-1]], dtype=tf.float32, name="actor_picked_actions")  # picked actions
        self.actor_targets = self.return_targets - self.critic_value
        targets = tf.expand_dims(self.actor_targets, -1)
        prev_layer = fork_layer
        with tf.variable_scope("private_to_actor"):
            for i, d in enumerate(self.actor_remaining_net_dim[:-1]):
                prev_layer = tl.fully_connected(prev_layer, d, scope="layer{}".format(i))
            self.mu = tl.fully_connected(prev_layer, self.actor_remaining_net_dim[-1], activation_fn=None, scope="mean_actions")
            # self.log_sigma_2 = tl.fully_connected(prev_layer, self.actor_remaining_net_dim[-1], activation_fn=None, scope="log_variance_actions")
            self.log_sigma_2 = tf.ones_like(self.mu) * np.log(0.01) * tf.cond(tf.greater(tf.random_uniform(shape=(), minval=0, maxval=1), tf.constant(0.9)), lambda: tf.constant(1.0), lambda: tf.constant(1e2))
        self.sigma_2 = tf.exp(self.log_sigma_2)
        self.sigma = tf.exp(0.5 * self.log_sigma_2)
        self.sample_action = tf.random_normal(shape=tf.shape(self.mu)) * self.sigma + self.mu
        self.probs = 1 / (tf.sqrt(2 * pi * self.sigma_2)) * tf.exp(-(self.actions - self.mu) ** 2 / (2 * self.sigma_2))
        self.log_probs = -0.5 * (log(2 * pi) + self.log_sigma_2 + (self.actions - self.mu) ** 2 / (self.sigma_2 + 1e-4))
        self.entropy = 0.5 * (1 + log(2 * pi) + self.log_sigma_2)
        self.entropy_mean = tf.reduce_mean(self.entropy, name="entropy_mean")
        self.actor_losses = -self.log_probs * targets - self.entropy_coef * self.entropy
        self.actor_loss = tf.reduce_sum(self.actor_losses, name="loss")
        # optimizer / summary
        self.global_rl_step = tf.Variable(0, dtype=tf.int32)
        self.global_both_step = tf.Variable(0, dtype=tf.int32)
        self.global_both_step_inc = self.global_both_step.assign_add(1)
        self.critic_optimizer = tf.train.AdamOptimizer(5e-5)
        # self.critic_optimizer = tf.train.AdamOptimizer(1e-3)
        self.critic_train_op = self.critic_optimizer.minimize(self.critic_loss, global_step=self.global_rl_step)
        # self.rl_optimizer = tf.train.AdamOptimizer(5e-5)
        # self.rl_train_op = self.rl_optimizer.minimize(0.00001 * self.actor_loss + self.critic_loss, global_step=self.global_rl_step)
        self.rl_optimizer = tf.train.AdamOptimizer(1e-3)
        self.rl_train_op = self.rl_optimizer.minimize(0.0000001 * self.actor_loss + self.critic_loss, global_step=self.global_rl_step)
        # Used in summaries
        self.rl_model_losses = tf.placeholder(shape=[None], dtype=tf.float32, name="rl_model_losses")
        self.rl_rewards = tf.placeholder(shape=[None], dtype=tf.float32, name="rl_rewards")
        # summaries
        names = ["arm1_arm2_left", "arm1_arm2_right", "ground_arm1_left", "ground_arm1_right"]
        mean_summaries, std_summaries, prob_summaries = [], [], []
        for i, name in zip(range(4), names):
            mean_summary = tf.summary.scalar("joints/{}_mean".format(name), self.mu[0, i])
            mean_summaries.append(mean_summary)
            std_summary = tf.summary.scalar("joints/{}_std".format(name), self.sigma[0, i])
            std_summaries.append(std_summary)
            prob_summary = tf.summary.scalar("joints/{}_prob".format(name), self.probs[0, i])
            prob_summaries.append(prob_summary)
        self.per_joint_summaries = mean_summaries + std_summaries + prob_summaries
        self.def_rl_summaries()

    def def_rl_summaries(self, summary_prefix=""):
        if summary_prefix != self._prev_summary_prefix_rl:
            self._prev_summary_prefix_rl = summary_prefix
            actor_loss_summary = tf.summary.scalar(summary_prefix + "/actor_loss", self.actor_loss)
            actor_stddev_sumary = tf.summary.scalar(summary_prefix + "/actor_stddev", tf.reduce_mean(self.sigma))
            critic_loss_summary = tf.summary.scalar(summary_prefix + "/critic_loss", self.critic_loss)
            self.rl_summary = tf.summary.merge([
                actor_loss_summary,
                critic_loss_summary,
                actor_stddev_sumary] + self.per_joint_summaries)
            model_loss_at_rl_time_summary = \
                tf.summary.scalar(summary_prefix + "/model_loss_at_rl_time", tf.reduce_mean(self.rl_model_losses))
            rl_reward_summary = tf.summary.scalar(summary_prefix + "/reward_at_rl_time", tf.reduce_mean(self.rl_rewards))
            self.rl_summary_stagewise = tf.summary.merge([
                model_loss_at_rl_time_summary,
                actor_loss_summary,
                critic_loss_summary,
                rl_reward_summary,
                actor_stddev_sumary] + self.per_joint_summaries)

    def def_model_summaries(self, summary_prefix=""):
        if summary_prefix != self._prev_summary_prefix_model:
            self._prev_summary_prefix_model = summary_prefix
            model_loss_at_model_time_summary = \
                tf.summary.scalar(summary_prefix + "/model_loss_at_model_time", self.model_loss)
            model_chance_loss_summary = tf.summary.scalar(summary_prefix + "/model_chance_loss", self.model_chance_loss)
            better_than_chance_summary = tf.summary.scalar(summary_prefix + "/better_than_chance", self.model_chance_loss - self.model_loss)
            model_reward_summary = tf.summary.scalar(summary_prefix + "/reward_at_model_time", tf.reduce_mean(self.rewards))
            self.model_summary = tf.summary.merge([
                model_loss_at_model_time_summary,
                model_reward_summary,
                better_than_chance_summary,
                model_chance_loss_summary])

    def get_model_state(self):
        return self.env.discrete_positions, self.env.discrete_speeds, self.env.discrete_target_positions

    def get_rl_state(self):
        return self.env.discrete_positions, self.env.discrete_speeds, self.env.discrete_target_positions

    def to_rl_feed_dict(self, states=None, actions=None, rewards=None, model_losses=None):
        # transforms the inputs into a feed dict for the actor
        feed_dict = {}
        if states is not None:
            np_rl_states = np.array(states)
            new_shape = (-1, np.prod(np_rl_states.shape[1:]))
            feed_dict[self.rl_inputs] = np.reshape(np_rl_states, new_shape)
        if actions is not None:
            np_actions = np.array(actions)
            feed_dict[self.actions] = np_actions
        if rewards is not None:
            # reverse pass through the rewards here...
            returns = self.rewards_to_return(rewards)
            feed_dict[self.rl_rewards] = rewards
            feed_dict[self.return_targets_not_bootstraped] = returns
        if model_losses is not None:
            feed_dict[self.rl_model_losses] = model_losses
        return feed_dict

    def to_model_feed_dict(self, states, targets=False):
        feed_dict = {}
        np_states = np.array(states)
        if targets:
            feed_dict[self.model_inputs] = np_states[:-1]
            feed_dict[self.model_targets] = np_states[1:, 0]
        else:
            feed_dict[self.model_inputs] = np_states
        return feed_dict


class JointJointAgentWorker(BaseJointAgentWorker):
    def define_model(self):
        net_dim = self.model_net_dim
        self.model_inputs = tf.placeholder(shape=[None, 3, 4, self.n_discrete], dtype=tf.float32, name="model_inputs")
        self.model_targets = tf.placeholder(shape=[None, 4, self.n_discrete], dtype=tf.float32, name="model_targets")
        input_positions = self.model_inputs[:, 0, :, :]
        flat_model_inputs = tf.reshape(self.model_inputs, (-1, 3 * 4 * self.n_discrete))
        flat_model_targets = tf.reshape(self.model_targets, (-1, 4 * self.n_discrete))
        prev_layer = flat_model_inputs
        with tf.variable_scope("model"):
            for i, d in enumerate(net_dim[1:]):
                activation_fn = tf.nn.relu if i < len(net_dim) - 2 else None
                prev_layer = tl.fully_connected(prev_layer, d, scope="layer{}".format(i), activation_fn=activation_fn)
        self.model_outputs = prev_layer
        self.model_losses = tf.reduce_mean((self.model_outputs - flat_model_targets) ** 2, axis=-1)
        self.model_loss = tf.reduce_mean(self.model_losses, name="loss")
        self.model_chance_loss = tf.reduce_mean((self.model_inputs[:, 0, :, :] - self.model_targets) ** 2)
        self.define_reward(**self.reward_params)
        self.global_model_step = tf.Variable(0, dtype=tf.int32)
        # optimizer / summary
        self.model_optimizer = tf.train.AdamOptimizer(1e-3)
        self.model_train_op = self.model_optimizer.minimize(self.model_loss, global_step=self.global_model_step)
        self.def_model_summaries()

    def define_net_dims(self):
        self.n_discrete = self.env._n_discrete
        self.model_net_dim = [self.n_discrete * 4 * 3, 600, 600, 600, self.n_discrete * 4]
        self.rl_shared_net_dim = [self.n_discrete * 4 * 3, 600]
        self.actor_remaining_net_dim = [4]
        self.critic_remaining_net_dim = [1]


class SquaredJointJointAgentWorker(BaseJointAgentWorker):
    def define_model(self):
        net_dim = self.model_net_dim
        self.model_inputs = tf.placeholder(shape=[None, 3, 4, self.n_discrete], dtype=tf.float32, name="model_inputs")
        self.model_targets = tf.placeholder(shape=[None, 4, self.n_discrete], dtype=tf.float32, name="model_targets")
        input_positions = self.model_inputs[:, 0, :, :]
        flat_model_inputs = tf.reshape(self.model_inputs, (-1, 3 * 4 * self.n_discrete))
        flat_model_targets = tf.reshape(self.model_targets, (-1, 4 * self.n_discrete))
        prev_layer = flat_model_inputs
        with tf.variable_scope("model"):
            for i, d in enumerate(net_dim[1:]):
                activation_fn = tf.nn.relu if i < len(net_dim) - 2 else None
                prev_layer = tl.fully_connected(prev_layer, d, scope="layer{}".format(i), activation_fn=activation_fn)
        self.model_outputs = prev_layer
        self.model_losses = tf.reduce_mean((self.model_outputs - flat_model_targets) ** 2, axis=-1)
        self.model_loss = tf.reduce_mean(self.model_losses, name="loss")
        self.model_chance_loss = tf.reduce_mean((self.model_inputs[:, 0, :, :] - self.model_targets) ** 2)
        prev_layer = flat_model_inputs
        with tf.variable_scope("model_squared"):
            for i, d in enumerate([100, 1]):
                activation_fn = tf.nn.relu if i < 1 else None
                prev_layer = tl.fully_connected(prev_layer, d, scope="layer{}".format(i), activation_fn=activation_fn)
        self.squared_loss = (self.model_losses - tf.squeeze(prev_layer)) ** 2
        self.define_reward(**self.reward_params)
        self.global_model_step = tf.Variable(0, dtype=tf.int32)
        self.total_loss = self.model_loss + tf.reduce_mean(self.squared_loss)

        # optimizer / summary
        self.model_optimizer = tf.train.AdamOptimizer(1e-3)
        self.model_train_op = self.model_optimizer.minimize(self.total_loss, global_step=self.global_model_step)
        self.def_model_summaries()

    def define_reward(self, coef=100):
        self.rewards = self.squared_loss * coef * (1 - self.discount_factor)

    def define_net_dims(self):
        self.n_discrete = self.env._n_discrete
        self.model_net_dim = [self.n_discrete * 4 * 3, 600, 600, 600, self.n_discrete * 4]
        self.rl_shared_net_dim = [self.n_discrete * 4 * 3]
        self.actor_remaining_net_dim = [600, 4]
        self.critic_remaining_net_dim = [600, 1]

    def def_model_summaries(self, summary_prefix=""):
        if summary_prefix != self._prev_summary_prefix_model:
            self._prev_summary_prefix_model = summary_prefix
            squared_summary = tf.summary.scalar(summary_prefix + "/pred_of_pred_error_error", self.squared_loss[0])
            model_loss_at_model_time_summary = \
                tf.summary.scalar(summary_prefix + "/model_loss_at_model_time", self.model_losses[0])
            model_chance_loss_summary = tf.summary.scalar(summary_prefix + "/model_chance_loss", self.model_chance_loss)
            better_than_chance_summary = tf.summary.scalar(summary_prefix + "/better_than_chance", self.model_chance_loss - self.model_loss)
            model_reward_summary = tf.summary.scalar(summary_prefix + "/reward_at_model_time", self.rewards[0])
            self.model_summary = tf.summary.merge([
                model_loss_at_model_time_summary,
                model_reward_summary,
                better_than_chance_summary,
                model_chance_loss_summary,
                squared_summary])


class SeparateJointAgentWorker(BaseJointAgentWorker):
    def define_model(self):
        net_dim = self.model_net_dim
        self.model_inputs = tf.placeholder(shape=[None, 3, 4, self.n_discrete], dtype=tf.float32, name="model_inputs")
        self.model_targets = tf.placeholder(shape=[None, 4, self.n_discrete], dtype=tf.float32, name="model_targets")
        splited_model_inputs = [tf.reshape(x, (-1, 3 * self.n_discrete)) for x in tf.unstack(self.model_inputs, 4, axis=2)]
        splited_model_targets = tf.unstack(self.model_targets, 4, axis=1)
        splited_model_outputs = []
        with tf.variable_scope("model"):
            for joint_id, inp in enumerate(splited_model_inputs):
                prev_layer = inp
                for i, d in enumerate(net_dim[1:]):
                    activation_fn = tf.nn.relu if i < len(net_dim) - 2 else None
                    prev_layer = tl.fully_connected(prev_layer, d, scope="joint{}_layer{}".format(joint_id, i), activation_fn=activation_fn)
                splited_model_outputs.append(prev_layer)
        self.model_outputs = tf.stack(splited_model_outputs, axis=1)
        self.model_losses = tf.reduce_mean((self.model_outputs - self.model_targets) ** 2, axis=[-2, -1])
        self.model_loss = tf.reduce_mean(self.model_losses, name="loss")
        self.model_chance_loss = tf.reduce_mean((self.model_inputs[:, 0, :, :] - self.model_targets) ** 2)
        self.define_reward(**self.reward_params)
        self.global_model_step = tf.Variable(0, dtype=tf.int32)
        # optimizer / summary
        self.model_optimizer = tf.train.AdamOptimizer(1e-3)
        self.model_train_op = self.model_optimizer.minimize(self.model_loss, global_step=self.global_model_step)
        self.def_model_summaries()

    def define_net_dims(self):
        self.n_discrete = self.env._n_discrete
        self.model_net_dim = [self.n_discrete * 3, 600, 600, 600, self.n_discrete]
        self.rl_shared_net_dim = [self.n_discrete * 4 * 3, 600]
        self.actor_remaining_net_dim = [4]
        self.critic_remaining_net_dim = [1]


class MinimizeJointAgentWorker(BaseJointAgentWorker):
    def define_reward(self, model_loss_converges_to):
        reward_scale = (1 - self.discount_factor) / model_loss_converges_to
        self.rewards = - reward_scale * self.model_losses


class MaximizeJointAgentWorker(BaseJointAgentWorker):
    def define_reward(self, model_loss_converges_to):
        reward_scale = (1 - self.discount_factor) / model_loss_converges_to
        self.rewards = reward_scale * self.model_losses


class TargetErrorJointAgentWorker(BaseJointAgentWorker):
    def define_reward(self, target_prediction_error):
        self.rewards = -tf.abs(self.model_losses - target_prediction_error) * (1 - self.discount_factor) / 0.04


class RangeErrorJointAgentWorker(BaseJointAgentWorker):
    def define_reward(self, range_mini, range_maxi):
        mean = (range_maxi + range_mini) / 2
        dist = (range_maxi - range_mini) / 2
        self.rewards = -tf.nn.relu(-(dist - tf.abs(self.model_losses - mean)) * (1 - self.discount_factor) / 0.04)


class Experiment:
    def __init__(self, n_parameter_servers, n_workers, WorkerCls, experiment_dir, args_env, args_worker, display_dpi=3):
        lock = filelock.FileLock("/home/wilmot/Documents/code/intrinsic_motivations_box2d/experiments/lock")
        lock.acquire()
        self.n_parameter_servers = n_parameter_servers
        self.n_workers = n_workers
        self.WorkerCls = WorkerCls
        self.experiment_dir = experiment_dir
        self.mktree()
        self.cluster = get_cluster(n_parameter_servers, n_workers)
        self.args_env, self.args_worker = args_env, list(args_worker)
        self.args_worker = [self.cluster, self.logdir] + self.args_worker
        self.args_env_display = list(args_env)
        self.args_env_display[5] = display_dpi
        pipes = [multiprocessing.Pipe(True) for i in range(n_workers)]
        self.here_pipes = [a for a, b in pipes]
        self.there_pipes = [b for a, b in pipes]
        self.here_worker_pipes = [a for a, b in pipes]
        self.here_display_pipes = []
        ### DEFINE PROCESSES ###
        self.tensorboard_process = None
        self.chromium_process = None
        self.parameter_servers_processes = [multiprocessing.Process(
            target=self.parameter_server_func,
            args=(i,),
            daemon=True)
            for i in range(self.n_parameter_servers)]
        self.workers_processes = [multiprocessing.Process(
            target=self.worker_func,
            args=(i,),
            daemon=True)
            for i in range(self.n_workers)]
        ### start all processes ###
        all_processes = self.parameter_servers_processes + self.workers_processes
        for p in all_processes:
            p.start()
        time.sleep(5)
        lock.release()

    def mktree(self):
        self.logdir = self.experiment_dir + "/log"
        self.checkpointsdir = self.experiment_dir + "/checkpoints"
        os.mkdir(self.experiment_dir)
        os.mkdir(self.logdir)
        os.mkdir(self.checkpointsdir)

    def parameter_server_func(self, task_index):
        server = tf.train.Server(self.cluster, "ps", task_index)
        server.join()

    def worker_func(self, task_index):
        env = environment.Environment(*self.args_env)
        worker = self.WorkerCls(task_index, self.there_pipes[task_index], env, *self.args_worker)
        worker.wait_for_variables_initialization()
        worker()

    def start_tensorboard(self):
        if self.tensorboard_process is not None and self.chromium_process is not None:
            if self.tensorboard_process.is_alive() or self.chromium_process.is_alive():
                print("restarting tensorboard")
                self.close_tensorboard()
        port = get_available_port()
        self.tensorboard_process = subprocess.Popen(["tensorboard", "--logdir", self.logdir, "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        time.sleep(2)
        self.chromium_process = subprocess.Popen(["chromium-browser", "http://localhost:{}".format(port)], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    def close_tensorboard(self):
        if self.tensorboard_process is not None and self.chromium_process is not None:
            self.tensorboard_process.terminate()
            self.chromium_process.terminate()

    def close_parameter_servers(self):
        for p in self.parameter_servers_processes:
            if p.is_alive():
                p.terminate()
        for p in self.parameter_servers_processes:
            while p.is_alive():
                time.sleep(0.1)

    def start_display_worker(self, sample=True):
        self.here_display_pipes.append(self.here_worker_pipes[-1])
        self.here_worker_pipes = self.here_worker_pipes[:-1]
        self.here_display_pipes[-1].send(("run_display", sample))

    def set_display_worker_idle(self):
        self.here_display_pipes[-1].send("done")  # quit run_display
        self.here_display_pipes[-1].recv()  # waiting
        self.here_worker_pipes.append(self.here_display_pipes[-1])
        self.here_display_pipes = self.here_display_pipes[:-1]

    def set_all_display_workers_idle(self):
        while len(self.here_display_pipes) > 0:
            self.set_display_worker_idle()

    def asynchronously_run_model(self, n_updates, summary_prefix):
        for p in self.here_worker_pipes:
            p.send(("run_model", n_updates, summary_prefix))
        for p in self.here_worker_pipes:
            p.recv()

    def asynchronously_run_reinforcement_learning(self, n_updates, summary_prefix, train_actor=True):
        for p in self.here_worker_pipes:
            p.send(("run_reinforcement_learning", n_updates, summary_prefix, train_actor))
        for p in self.here_worker_pipes:
            p.recv()

    def asynchronously_run_both(self, n_updates, summary_prefix, train_actor=True):
        for p in self.here_worker_pipes:
            p.send(("run_all", n_updates, summary_prefix, train_actor))
        for p in self.here_worker_pipes:
            p.recv()

    def save_model(self, name):
        path = self.checkpointsdir + "/{}/".format(name)
        os.mkdir(path)
        self.here_worker_pipes[0].send(("save", path))
        print(self.here_worker_pipes[0].recv())

    def restore_model(self, path):
        self.here_worker_pipes[0].send(("restore", path))
        print(self.here_worker_pipes[0].recv())
        # for p in self.here_worker_pipes:
        #     p.send(("restore", path))
        #     print(p.recv())

    def close_workers(self):
        for p in self.here_worker_pipes:
            p.send("done")

    def close(self):
        self.close_tensorboard()
        self.set_all_display_workers_idle()
        self.close_workers()
        self.close_parameter_servers()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


# class Experiment:
#     def __init__(self, n_parameter_servers, n_workers, WorkerCls, experiment_dir, args_env, args_worker, display_dpi=3):
#         self.n_parameter_servers = n_parameter_servers
#         self.n_workers = n_workers
#         self.WorkerCls = WorkerCls
#         self.cluster = get_cluster(n_parameter_servers, n_workers)
#         self.parameter_servers_processes = []
#         self.workers_processes = []
#         self.workers_events = []
#         self.experiment_dir = experiment_dir
#         self.mktree()
#         self.args_env, self.args_worker = args_env, list(args_worker)
#         self.args_worker = [self.cluster, self.logdir] + self.args_worker
#         self.args_env_display = list(args_env)
#         self.args_env_display[5] = display_dpi
#         self._define_tensorboard_process()
#         self.display_process = None
#
#         # Define processes:
#     def _define_tensorboard_process(self):
#         self.tensorboard_process = multiprocessing.Process(target=self.tensorboard_func)
#
#     def _define_display_process(self, sample=True):
#         self.display_event = multiprocessing.Event()
#         self.display_process = multiprocessing.Process(target=self.worker_display_func, args=(sample,), daemon=True)
#
#     def _define_parameter_server_processes(self):
#         for i in range(self.n_parameter_servers):
#             p = multiprocessing.Process(target=self.parameter_server_func, args=(i,), daemon=True)
#             self.parameter_servers_processes.append(p)
#
#     def _define_workers_model_processes(self, n_updates, summary_prefix):
#         for i in range(self.n_workers):
#             p = multiprocessing.Process(target=self.worker_model_func, args=(i, n_updates, summary_prefix), daemon=True)
#             self.workers_processes.append(p)
#             self.workers_events.append(multiprocessing.Event())
#
#     def _define_workers_rl_processes(self, n_updates, summary_prefix, train_actor=True):
#         for i in range(self.n_workers):
#             p = multiprocessing.Process(target=self.worker_rl_func, args=(i, n_updates, summary_prefix, train_actor), daemon=True)
#             self.workers_processes.append(p)
#             self.workers_events.append(multiprocessing.Event())
#
#     def _define_workers_both_processes(self, n_updates, summary_prefix, train_actor=True):
#         for i in range(self.n_workers):
#             p = multiprocessing.Process(target=self.worker_both_func, args=(i, n_updates, summary_prefix, train_actor), daemon=True)
#             self.workers_processes.append(p)
#             self.workers_events.append(multiprocessing.Event())
#
#     def _define_workers_save_processes(self, path):
#         for i in range(self.n_workers):
#             p = multiprocessing.Process(target=self.worker_save_func, args=(i, path), daemon=True)
#             self.workers_processes.append(p)
#             self.workers_events.append(multiprocessing.Event())
#
#     def _define_workers_restore_processes(self, path):
#         for i in range(self.n_workers):
#             p = multiprocessing.Process(target=self.worker_restore_func, args=(i, path), daemon=True)
#             self.workers_processes.append(p)
#             self.workers_events.append(multiprocessing.Event())
#
#     def worker_display_func(self, sample=True):
#         env = environment.Environment(*self.args_env_display)
#         worker = self.WorkerCls(self.n_workers - 1, self.display_event, env, *self.args_worker)
#         worker.wait_for_variables_initialization()
#         worker.run_display(sample)
#
#     def tensorboard_func(self):
#         port = get_available_port()
#         p1 = multiprocessing.Process(target=tensorboard_server_func, args=(self.logdir, port), daemon=True)
#         p1.start()
#         time.sleep(2)
#         p2 = multiprocessing.Process(target=chromium_func, args=(port,), daemon=True)
#         p2.start()
#         p2.join()
#         terminate_process_safe(p1)
#
#     def parameter_server_func(self, task_index):
#         server = tf.train.Server(self.cluster, "ps", task_index)
#         server.join()
#
#     def worker_model_func(self, task_index, n_updates, summary_prefix):
#         env = environment.Environment(*self.args_env)
#         worker = self.WorkerCls(task_index, self.workers_events[task_index], env, *self.args_worker, summary_prefix=summary_prefix)
#         worker.wait_for_variables_initialization()
#         worker.run_model(n_updates)
#
#     def worker_rl_func(self, task_index, n_updates, summary_prefix, train_actor=True):
#         env = environment.Environment(*self.args_env)
#         worker = self.WorkerCls(task_index, self.workers_events[task_index], env, *self.args_worker, summary_prefix=summary_prefix)
#         worker.wait_for_variables_initialization()
#         worker.run_reinforcement_learning(n_updates, train_actor)
#
#     def worker_both_func(self, task_index, n_updates, summary_prefix, train_actor=True):
#         env = environment.Environment(*self.args_env)
#         worker = self.WorkerCls(task_index, self.workers_events[task_index], env, *self.args_worker, summary_prefix=summary_prefix)
#         worker.wait_for_variables_initialization()
#         worker.run_all(n_updates, train_actor)
#
#     def worker_save_func(self, task_index, path):
#         env = environment.Environment(*self.args_env)
#         worker = self.WorkerCls(task_index, self.workers_events[task_index], env, *self.args_worker)
#         worker.wait_for_variables_initialization()
#         worker.save(path)
#
#     def worker_restore_func(self, task_index, path):
#         env = environment.Environment(*self.args_env)
#         worker = self.WorkerCls(task_index, self.workers_events[task_index], env, *self.args_worker)
#         worker.wait_for_variables_initialization()
#         worker.restore(path)
#
#     def __enter__(self):
#         self.start_parameter_servers()
#         return self
#
#     def __exit__(self, type, value, traceback):
#         self.close()
#
#     def mktree(self):
#         self.logdir = self.experiment_dir + "/log"
#         self.checkpointsdir = self.experiment_dir + "/checkpoints"
#         os.mkdir(self.experiment_dir)
#         os.mkdir(self.logdir)
#         os.mkdir(self.checkpointsdir)
#
#     def start_tensorboard(self):
#         if not self.tensorboard_process.is_alive():
#             self.tensorboard_process.start()
#
#     def close_tensorboard(self):
#         if self.tensorboard_process.is_alive():
#             terminate_process_safe(self.tensorboard_process)
#
#     def start_parameter_servers(self):
#         self._define_parameter_server_processes()
#         for p in self.parameter_servers_processes:
#             if not p.is_alive():
#                 p.start()
#
#     def close_parameter_servers(self):
#         for p in self.parameter_servers_processes:
#             if p.is_alive():
#                 p.terminate()
#         for p in self.parameter_servers_processes:
#             while p.is_alive():
#                 time.sleep(0.1)
#
#     def start_display_worker(self, sample=True):
#         if self.display_process is None or not self.display_process.is_alive():
#             self._define_display_process(sample)
#             self.display_process.start()
#
#     def close_display_worker(self):
#         if self.display_process is not None and self.display_process.is_alive():
#             self.display_event.set()
#             while self.display_process.is_alive():
#                 time.sleep(0.1)
#
#     def asynchronously_run_model(self, n_updates, summary_prefix):
#         self._define_workers_model_processes(n_updates, summary_prefix)
#         self._start_workers_processes()
#
#     def asynchronously_run_reinforcement_learning(self, n_updates, summary_prefix, train_actor=True):
#         self._define_workers_rl_processes(n_updates, summary_prefix, train_actor=train_actor)
#         self._start_workers_processes()
#
#     def asynchronously_run_both(self, n_updates, summary_prefix, train_actor=True):
#         self._define_workers_both_processes(n_updates, summary_prefix, train_actor=train_actor)
#         self._start_workers_processes()
#
#     def save_model(self, name):
#         # path = os.path.abspath(self.checkpointsdir + "/{}/".format(name))
#         path = self.checkpointsdir + "/{}/".format(name)
#         os.mkdir(path)
#         self._define_workers_save_processes(path)
#         self._start_workers_processes()
#
#     def restore_model(self, path):
#         self._define_workers_restore_processes(path)
#         self._start_workers_processes()
#
#     def _start_workers_processes(self):
#         if self.display_process is not None and self.display_process.is_alive():
#             procs = self.workers_processes[:-1]
#             events = self.workers_events[:-1]
#         else:
#             procs = self.workers_processes
#             events = self.workers_events
#         blockrun(procs, events)
#         self.workers_events = []
#         self.workers_processes = []
#
#     def close(self):
#         # self.close_tensorboard()
#         self.close_display_worker()
#         self.close_parameter_servers()


# def blockrun(procs, events):
#     for p in procs:
#         p.start()
#     for e in events:
#         e.wait()
#     for p in procs:
#         p.terminate()
#     for p in procs:
#         while p.is_alive():
#             time.sleep(0.1)


# def terminate_process_safe(p):
#     p.terminate()
#     while p.is_alive():
#         time.sleep(0.1)


if __name__ == "__main__":
    skin_order = [
        ("Arm1_Left", 0),
        ("Arm2_Left", 0),
        ("Arm2_Left", 1),
        ("Arm2_Left", 2),
        ("Arm1_Left", 2),
        ("Arm1_Right", 0),
        ("Arm2_Right", 0),
        ("Arm2_Right", 1),
        ("Arm2_Right", 2),
        ("Arm1_Right", 2)]
    skin_resolution = 12
    xlim = [-20.5, 20.5]
    ylim = [-13.5, 13.5]
    json_model = "../models/two_arms.json"
    dpi = 1
    dt = 1 / 150
    n_discrete = 32

    discount_factor = 0.0
    env_step_length = 45
    sequence_length = 128  # 1024  # 64
    logdir = TemporaryDirectory()

    N_WORKERS = 16
    N_PARAMETER_SERVERS = 8

    cluster = get_cluster(N_PARAMETER_SERVERS, N_WORKERS)
    args_env = (json_model, skin_order, skin_resolution, xlim, ylim, dpi, dt, n_discrete)
    # args_worker = (cluster, logdir.name, discount_factor, env_step_length, sequence_length)
    args_worker = (discount_factor, env_step_length, sequence_length)

    with Experiment(
            N_PARAMETER_SERVERS, N_WORKERS, JointAgentWorker,
            "../experiments/sequence_128_env_step_45_df_000_max_pred_repetition_1", args_env, args_worker, display_dpi=3) as experiment:
        # experiment.start_display_worker()
        # experiment.start_tensorboard()
        experiment.save_model("initial")
        for i in range(20):
            summary_prefix = "stage_{}".format(i)
            experiment.asynchronously_run_model(4000 if i == 0 else 1000, summary_prefix)
            experiment.save_model("after_model_{}".format(i))
            experiment.asynchronously_run_reinforcement_learning(500, summary_prefix, train_actor=False)
            experiment.asynchronously_run_reinforcement_learning(2500, summary_prefix, train_actor=True)
            experiment.save_model("after_rl_{}".format(i))
