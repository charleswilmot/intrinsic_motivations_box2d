import sys
import pickle
from discretization import np_discretization_reward
from scipy import special
from tempfile import TemporaryDirectory
import png
import environment
from replay_buffer import Buffer
from goal_library import GoalLibrary
import socket
import multiprocessing
import subprocess
import numpy as np
from numpy import pi, log
import tensorflow as tf
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import layers
import time
import viewer
import os
import filelock
from imageio import get_writer


def lrelu(x):
    alpha = 0.2
    return tf.nn.relu(x) * (1 - alpha) + x * alpha


def exponential_moving_stats(ten, alpha):
    mean, var = tf.nn.moments(ten, axes=0)
    std = tf.sqrt(var)
    moving_mean = tf.Variable(tf.zeros_like(mean))
    moving_mean_assign = moving_mean.assign(alpha * moving_mean + (1 - alpha) * mean)
    moving_std = tf.Variable(tf.ones_like(std))
    moving_std_assign = moving_std.assign(alpha * moving_std + (1 - alpha) * std)
    cond = tf.less(tf.shape(ten)[0], 2)
    moving_mean_cond = tf.cond(cond, lambda: moving_mean, lambda: moving_mean_assign)
    moving_std_cond = tf.cond(cond, lambda: moving_std, lambda: moving_std_assign)
    return moving_mean_cond, moving_std_cond


def normalize(ten, alpha):
    mean, std = exponential_moving_stats(ten, alpha)
    return (ten - mean) / (std + 1e-5)


def get_cluster(n_parameter_servers, n_workers):
    spec = {}
    port = get_available_port(2222)
    for i in range(n_parameter_servers):
        if "ps" not in spec:
            spec["ps"] = []
        port = get_available_port(port + 1)
        spec["ps"].append("localhost:{}".format(port))
    for i in range(n_workers):
        if "worker" not in spec:
            spec["worker"] = []
        port = get_available_port(port + 1)
        spec["worker"].append("localhost:{}".format(port))
    return tf.train.ClusterSpec(spec)


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def get_available_port(start_port=6006):
    port = start_port
    while is_port_in_use(port):
        port += 1
    return port


class MergeDenseNet(layers.Layer):
    def __init__(self, n_out):
        super().__init__()
        self.n_out = n_out
        self.layer1 = layers.Dense(200, activation=tf.nn.leaky_relu)
        self.layer2 = layers.Dense(200, activation=tf.nn.leaky_relu)
        self.layer3 = layers.Dense(200, activation=tf.nn.tanh)
        self.layer4 = layers.Dense(n_out, activation=None)


class ConcatDenseNet(MergeDenseNet):
    def call(self, inputs):
        tensor, param = inputs
        tensor = self.layer1(tensor)
        tensor = tf.concat([tensor, param], axis=-1)
        tensor = self.layer2(tensor)
        tensor = tf.concat([tensor, param], axis=-1)
        tensor = self.layer3(tensor)
        tensor = tf.concat([tensor, param], axis=-1)
        tensor = self.layer4(tensor)
        return tensor

class RegularDenseNet(MergeDenseNet):
    def call(self, inputs):
        tensor, param = inputs
        tensor = self.layer1(tensor)
        tensor = self.layer2(tensor)
        tensor = self.layer3(tensor)
        tensor = self.layer4(tensor)
        return tensor

class AffineDenseNet(MergeDenseNet):
    def __init__(self, n_out, n_hidden_units=200):
        super().__init__(n_out)
        self.layer1_1 = layers.Dense(n_hidden_units, activation=tf.nn.leaky_relu)
        self.layer1_2 = layers.Dense(200 * 2, activation=tf.nn.leaky_relu)
        self.layer2_1 = layers.Dense(n_hidden_units, activation=tf.nn.leaky_relu)
        self.layer2_2 = layers.Dense(200 * 2, activation=tf.nn.leaky_relu)
        self.layer3_1 = layers.Dense(n_hidden_units, activation=tf.nn.leaky_relu)
        self.layer3_2 = layers.Dense(200 * 2, activation=tf.nn.leaky_relu)
        self.layer4_1 = layers.Dense(n_hidden_units)
        self.layer4_2 = layers.Dense(self.n_out * 2)

    def call(self, inputs):
        tensor, param = inputs
        tensor = self.layer1(tensor)
        scale_shift = self.layer1_1(param)
        scale_shift = self.layer1_2(scale_shift)
        scale, shift = scale_shift[..., :200], scale_shift[..., 200:]
        tensor = tensor * scale + shift
        tensor = self.layer2(tensor)
        scale_shift = self.layer2_1(param)
        scale_shift = self.layer2_2(scale_shift)
        scale, shift = scale_shift[..., :200], scale_shift[..., 200:]
        tensor = tensor * scale + shift
        tensor = self.layer3(tensor)
        scale_shift = self.layer3_1(param)
        scale_shift = self.layer3_2(scale_shift)
        scale, shift = scale_shift[..., :200], scale_shift[..., 200:]
        tensor = tensor * scale + shift
        tensor = self.layer4(tensor)
        scale_shift = self.layer4_1(param)
        scale_shift = self.layer4_2(scale_shift)
        scale, shift = scale_shift[..., :self.n_out], scale_shift[..., self.n_out:]
        tensor = tensor * scale + shift
        return tensor


class Worker:
    def __init__(self, task_index, pipe, cluster, logdir, goal_library, conf):
        self.task_index = task_index
        self.cluster = cluster
        self._n_workers = self.cluster.num_tasks("worker") - 1
        self.job_name = "worker"
        self.server = tf.train.Server(cluster, self.job_name, task_index)
        self.name = "/job:{}/task:{}".format(self.job_name, task_index)
        self.device = tf.train.replica_device_setter(worker_device=self.name, cluster=cluster)
        self.logdir = logdir
        self.conf = conf
        self.env = environment.Environment.from_conf(self.conf.environment_conf)
        self.discount_factor = self.conf.worker_conf.discount_factor
        self.sequence_length = self.conf.worker_conf.sequence_length
        self.critic_lr = self.conf.worker_conf.critic_learning_rate
        self.epsilon_init = self.conf.worker_conf.epsilon_init
        self.epsilon_decay = self.conf.worker_conf.epsilon_decay
        self.softmax_temperature = self.conf.goal_library_conf.softmax_temperature
        self.min_reaching_prob = self.conf.goal_library_conf.min_reaching_prob
        self.replay_buffer_size = self.conf.worker_conf.buffer_size
        self.updates_per_episode = self.conf.worker_conf.updates_per_episode
        self.her_strategy = self.conf.worker_conf.her_strategy
        self.n_actions = self.conf.worker_conf.n_actions
        self.actions_ground_to_arm = np.linspace(-3.14, 3.14, self.n_actions)
        self.actions_arm_to_arm = np.linspace(-3.14, 3.14, self.n_actions)
        self.parametrization_type = self.conf.worker_conf.parametrization_type
        self.pipe = pipe
        self.replay_buffer = Buffer(self.replay_buffer_size)
        p, s, t = self.get_state()
        self._p_size, self._s_size, self._t_size = p.shape, s.shape, t.shape

        self.goal_library = goal_library
        self.goal_library_size = self.goal_library.library_size
        self.goal_library_ema_speed = self.goal_library.ema_speed
        self.define_networks()
        # graph = tf.get_default_graph() if task_index == 0 else None
        graph = None
        self.summary_writer = tf.summary.FileWriter(self.logdir + "/worker{}".format(task_index), graph=graph)
        self.saver = tf.train.Saver()
        self._initializer = tf.global_variables_initializer()
        self._report_non_initialized = tf.report_uninitialized_variables()
        self.sess = tf.Session(target=self.server.target)

    def initialize(self):
        self.sess.run(self._initializer)
        print("{}  variables initialized".format(self.name))

    def actions_dict_from_indices(self, actions):
        return {
            "Arm1_to_Arm2_Left": self.actions_arm_to_arm[actions[0]],
            "Ground_to_Arm1_Left": self.actions_ground_to_arm[actions[1]],
            "Arm1_to_Arm2_Right": self.actions_arm_to_arm[actions[2]],
            "Ground_to_Arm1_Right": self.actions_ground_to_arm[actions[3]]
        }

    def wait_for_variables_initialization(self):
        while len(self.sess.run(self._report_non_initialized)) > 0:
            print("{}  waiting for variable initialization...".format(self.name))
            time.sleep(1)

    def __call__(self):
        self.pipe.send("{} going idle".format(self.name))
        cmd = self.pipe.recv()
        while not cmd == "done":
            try:
                print("{} got command {}".format(self.name, cmd))
                self.__getattribute__(cmd[0])(*cmd[1:])
                cmd = self.pipe.recv()
            except KeyboardInterrupt as e:
                print("{} caught a keyboard interrupt".format(self.name))
            except Exception as e:
                print("{} caught exception in worker:".format(self.name))
                raise e

    def save(self, path):
        iteration = self.sess.run(self.global_step)
        path += "/{:09d}".format(iteration)
        os.mkdir(path)
        save_path = self.saver.save(self.sess, path + "/network.ckpt")
        self.pipe.send("{} saved model to {}".format(self.name, save_path))

    def restore(self, path):
        self.saver.restore(self.sess, os.path.normpath(path + "/network.ckpt"))
        self.pipe.send("{} variables restored from {}".format(self.name, path))

    def run_training(self, n_updates, train_actor=True):
        global_step = self.sess.run(self.global_step)
        n_updates += global_step
        while global_step < n_updates - self._n_workers:
            # Collect some experience
            trajectory = self.run_n_steps()
            # Place in buffer
            self.replay_buffer.incorporate(trajectory)
            if self.her_strategy.lower() == "none":
                pass
            elif self.her_strategy.lower() == "first_contact":
                for tactile in trajectory["state_tactile"]:
                    if np.sum(tactile) > 0:
                        faked_goals = np.repeat(tactile[np.newaxis], self.sequence_length, 0)
                        faked_rewards = np_discretization_reward(trajectory["state_tactile"], faked_goals) * (1 - self.discount_factor)
                        faked_trajectory = {
                            "state_position": trajectory["state_position"],
                            "state_speed": trajectory["state_speed"],
                            "state_tactile": trajectory["state_tactile"],
                            "actions": trajectory["actions"],
                            "goals": faked_goals,
                            "rewards": faked_rewards
                        }
                        self.replay_buffer.incorporate(faked_trajectory)
                        break
            elif self.her_strategy.lower() == "all_contacts":
                goals_done = []
                for tactile in trajectory["state_tactile"]:
                    if np.sum(tactile) > 0:
                        # check if a similar fake goal has already been added to the replay buffer
                        if len(goals_done) > 0:
                            goals = np.repeat(tactile[np.newaxis], len(goals_done), 0)
                            rewards = np_discretization_reward(goals, goals_done)
                        if len(goals_done) == 0 or (rewards != 1).all():
                            goals_done.append(tactile)
                            faked_goals = np.repeat(tactile[np.newaxis], self.sequence_length, 0)
                            faked_rewards = np_discretization_reward(trajectory["state_tactile"], faked_goals) * (1 - self.discount_factor)
                            faked_trajectory = {
                                "state_position": trajectory["state_position"],
                                "state_speed": trajectory["state_speed"],
                                "state_tactile": trajectory["state_tactile"],
                                "actions": trajectory["actions"],
                                "goals": faked_goals,
                                "rewards": faked_rewards
                            }
                            self.replay_buffer.incorporate(faked_trajectory)
            else:
                raise NotImplementedError("HER strategy not recognized ({})".format(self.her_strategy))
            # Update the global networks
            trajectories = self.replay_buffer.batch(self.updates_per_episode)
            stacked_trajectories = {
                "state_position": np.stack([x["state_position"] for x in trajectories]),
                "state_speed": np.stack([x["state_speed"] for x in trajectories]),
                "state_tactile": np.stack([x["state_tactile"] for x in trajectories]),
                "actions": np.stack([x["actions"] for x in trajectories]),
                "goals": np.stack([x["goals"] for x in trajectories]),
                "rewards": np.stack([x["rewards"] for x in trajectories])
            }
            global_step = self.update_reinforcement_learning(stacked_trajectories)
            if global_step >= n_updates - self._n_workers:
                break
        self.pipe.send("{} going IDLE".format(self.name))

    def run_display(self, training=True):
        win = viewer.SkinAgentWindow(self.discount_factor, return_lookback=50)
        i = 0
        while not self.pipe.poll():
            self.run_n_display_steps(win, training)
            i = (i + 1) % 2
            if i == 0:
                self.randomize_env()
        win.close()
        self.pipe.recv()  # done
        self.pipe.send("{} (display) going IDLE".format(self.name))

    def print_goal_library(self):
        self.goal_library.print_with_header("{} goal library:".format(self.name))
        self.pipe.send("{} printed goal library".format(self.name))

    # def run_video(self, path, n_sequences, training=True):
    #     start_index = 0
    #     with TemporaryDirectory() as tmppath:
    #         while not os.path.isdir(tmppath):
    #             time.sleep(0.1)
    #         for i in range(n_sequences):
    #             start_index = self.run_n_video_steps(tmppath, start_index, training=training)
    #         os.system("ffmpeg -loglevel panic -r 24 -i {}/frame_%06d.png -vcodec mpeg4 -b 100000 -y {}".format(tmppath, path))
    #     self.pipe.send("{} saved video under {}".format(self.name, path))

    def take_goal_library_snapshot(self):
        iteration = self.sess.run(self.global_step)
        self.goal_library.take_snapshot(iteration)
        self.pipe.send("{} took a snapshot of the goal library".format(self.name))

    def restore_goal_library(self, path):
        self.goal_library.restore(path)
        self.pipe.send("{} restored the goal library".format(self.name))

    def dump_goal_library(self, path):
        self.goal_library.dump(path + "/worker_{}.pkl".format(self.task_index))
        self.pipe.send("{} saved library under {}".format(self.name, path))

    def save_contact_logs(self, name):
        path = self.logdir + "/worker{}/contacts_{}.pkl".format(self.task_index, name)
        self.env.save_contact_logs(path)
        self.pipe.send("{} saved contact logs under {}".format(self.name, path))

    def get_action(self, goal):
        goals = goal[np.newaxis, np.newaxis]
        feed_dict = self.to_feed_dict(states=self.get_state(True), goals=goals)
        action = self.sess.run(self.sampled_actions_indices, feed_dict=feed_dict)[0, 0]
        return action

    def populate_goal_library(self):
        no_goal = np.zeros(self._t_size)
        while(self.goal_library._n_goals < self.goal_library_size):
            before = self.goal_library._n_goals
            action = np.random.randint(0, self.n_actions, 4)
            # self.env.set_positions(self.actions_dict_from_indices(action))
            self.env.set_speeds(self.actions_dict_from_indices(action))
            self.env.env_step()
            self.goal_library.register_goal(
                observed_goal=self.env.tactile,
                pursued_goal=no_goal,
                vision=self.env.vision)  # point is to have a snapshot of the contact stored in the library
            after = self.goal_library._n_goals
            if before != after:
                print("{} populating goal library: {: 4d} / {: 4d}".format(self.name, after, self.goal_library_size))
        self.pipe.send("{} goal library full".format(self.name))

    def run_n_steps(self):
        state_position = np.zeros((self.sequence_length, ) + self._p_size)
        state_speed = np.zeros((self.sequence_length, ) + self._s_size)
        state_tactile = np.zeros((self.sequence_length, ) + self._t_size)
        # goal_index, goal = self.goal_library.select_goal_learning_potential(self.softmax_temperature)
        goal_index, goal = self.goal_library.select_goal_uniform_reachable_only(min_prob=self.min_reaching_prob)
        goals = np.repeat(goal[np.newaxis], self.sequence_length, 0)
        actions = np.zeros((self.sequence_length, 4))
        rewards = np.zeros((self.sequence_length, ))
        for iteration in range(self.sequence_length):
            # get action
            action = self.get_action(goal)
            actions[iteration] = action
            # set action
            # self.env.set_positions(self.actions_dict_from_indices(action))
            self.env.set_speeds(self.actions_dict_from_indices(action))
            # get states
            state_position[iteration], state_speed[iteration], state_tactile[iteration] = self.get_state()
            reward = self.goal_library.register_goal(
                observed_goal=state_tactile[iteration],
                pursued_goal=goal,
                vision=self.env.vision) * (1 - self.discount_factor)  # point is to have a snapshot of the contact stored in the library
            rewards[iteration] = reward  # different way of computing the reward (numpy vs tensorflow) see run_n_display_steps
            # run environment step
            self.env.env_step()
        self.randomize_env()
        trajectory = {
            "state_position": state_position,
            "state_speed": state_speed,
            "state_tactile": state_tactile,
            "actions": actions,
            "goals": goals,
            "rewards": rewards
        }
        return trajectory

    def randomize_env(self, n=None, _answer=False):
        n = 5 if n is None else n
        for i in range(n):
            action = np.random.randint(0, self.n_actions, 4)
            # self.env.set_positions(self.actions_dict_from_indices(action))
            self.env.set_speeds(self.actions_dict_from_indices(action))
            self.env.env_step()
        if _answer:
            self.pipe.send("{} applied {} random actions".format(self.name, n))

    def run_n_display_steps(self, win, training=True):
        rewards = np.zeros(self.sequence_length)
        # goal_index, goal = self.goal_library.select_goal_learning_potential(self.goal_temperature)
        goal_index, goal = self.goal_library.select_goal_uniform_learned_only(min_prob=0.07)
        reaching_probability = self.goal_library.goal_info(goal_index)["r|p"]
        if training:
            action_fetches = self.sampled_actions_indices
            critic_fetches = self.sampled_q_value
        else:
            action_fetches = self.greedy_actions_indices
            critic_fetches = self.greedy_q_value
        fetches = [action_fetches, critic_fetches]
        for i in range(self.sequence_length):
            feed_dict = self.to_feed_dict(states=self.get_state(True), goals=goal[np.newaxis, np.newaxis])
            action, predicted_returns = self.sess.run(fetches, feed_dict=feed_dict)
            action = action[0, 0]
            rewards[i] = np_discretization_reward(self.env.tactile, goal) * (1 - self.discount_factor)
            # set positions in env
            # self.env.set_positions(self.actions_dict_from_indices(action))
            self.env.set_speeds(self.actions_dict_from_indices(action))
            # run action in env
            self.env.env_step()
            # get current vision
            vision = self.env.vision
            tactile_true = self.env.tactile
            current_reward = rewards[i]
            predicted_return = np.max(predicted_returns)
            # display
            win(vision, tactile_true, goal, current_reward, predicted_return, reaching_probability=reaching_probability)

    def save_vision_related_to_goals(self, goaldir):
        # func : i, data -> action
        self.goal_library.save_vision(goaldir + "/worker{}/".format(self.task_index))
        self.pipe.send("{} saved vision related to the goals in the goals library".format(self.name))

    def save_video(self, path, goal_index, n_frames=2000, training=False):
        # goal_index, goal = self.goal_library.select_goal_learning_potential(self.goal_temperature)
        win = viewer.SkinAgentWindow(self.discount_factor, return_lookback=50, show=False)
        win.set_return_lim([-0.1, 1.1])
        width, height = win.fig.canvas.get_width_height()
        goal_info = self.goal_library.goal_info(goal_index)
        goal = goal_info["intensities"]
        reaching_probability = goal_info["r|p"]
        goals = goal[np.newaxis, :]
        if training:
            action_fetches = self.sampled_actions_indices
            critic_fetches = self.sampled_q_value
        else:
            action_fetches = self.greedy_actions_indices
            critic_fetches = self.greedy_q_value
        fetches = [action_fetches, critic_fetches]
        with get_writer(path + "/{:04d}.avi".format(goal_index), fps=25, format="avi", quality=5) as writer:
            for i in range(n_frames):
                feed_dict = self.to_feed_dict(states=self.get_state(True), goals=goal[np.newaxis, np.newaxis])
                action, predicted_returns = self.sess.run(fetches, feed_dict=feed_dict)
                action = action[0, 0]
                # set positions in env
                action_dict = self.actions_dict_from_indices(action)
                # self.env.set_positions(action_dict)
                self.env.set_speeds(action_dict)
                # run action in env
                self.env.env_step()
                # get current vision
                vision = self.env.vision
                tactile_true = self.env.tactile
                current_reward = np_discretization_reward(tactile_true, goal) * (1 - self.discount_factor)
                predicted_return = np.max(predicted_returns)
                # display
                win(vision, tactile_true, goal, current_reward, predicted_return, reaching_probability=reaching_probability)
                ############
                # methode 1: ( https://matplotlib.org/3.1.1/gallery/animation/frame_grabbing_sgskip.html )
                # writer.grab_frame()
                ############
                # methode 2: ( http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure )
                frame = np.fromstring(win.fig.canvas.tostring_argb(), dtype=np.uint8).reshape(height, width, 4)
                frame = frame[:, :, 1:]               # if we discard the alpha chanel
                writer.append_data(frame)
        self.pipe.send("{} saved video under {}".format(self.name, path))

    def save_video_all_goals(self, path, n_frames=50, training=False):
        win = viewer.SkinAgentWindow(self.discount_factor, return_lookback=50, show=False)
        win.set_return_lim([-0.1, 1.1])
        width, height = win.fig.canvas.get_width_height()
        with get_writer(path + "/bragging.mp4", fps=25, format="mp4", quality=5) as writer:
            for goal_index in np.argsort(self.goal_library.goal_array["r|p"])[::-1]:
                goal_info = self.goal_library.goal_info(goal_index)
                goal = goal_info["intensities"]
                reaching_probability = goal_info["r|p"]
                goals = goal[np.newaxis, :]
                if training:
                    action_fetches = self.sampled_actions_indices
                    critic_fetches = self.sampled_q_value
                else:
                    action_fetches = self.greedy_actions_indices
                    critic_fetches = self.greedy_q_value
                fetches = [action_fetches, critic_fetches]
                print("{} goal index: {: 3d}    reaching probability: {:.2f}".format(self.name, goal_index, reaching_probability))
                for i in range(n_frames):
                    feed_dict = self.to_feed_dict(states=self.get_state(True), goals=goal[np.newaxis, np.newaxis])
                    action, predicted_returns = self.sess.run(fetches, feed_dict=feed_dict)
                    action = action[0, 0]
                    # set positions in env
                    action_dict = self.actions_dict_from_indices(action)
                    # self.env.set_positions(action_dict)
                    self.env.set_speeds(action_dict)
                    # run action in env
                    self.env.env_step()
                    # get current vision
                    vision = self.env.vision
                    tactile_true = self.env.tactile
                    current_reward = np_discretization_reward(tactile_true, goal) * (1 - self.discount_factor)
                    predicted_return = np.max(predicted_returns)
                    # display
                    win(vision, tactile_true, goal, current_reward, predicted_return, reaching_probability=reaching_probability)
                    ############
                    # methode 1: ( https://matplotlib.org/3.1.1/gallery/animation/frame_grabbing_sgskip.html )
                    # writer.grab_frame()
                    ############
                    # methode 2: ( http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure )
                    frame = np.fromstring(win.fig.canvas.tostring_argb(), dtype=np.uint8).reshape(height, width, 4)
                    frame = frame[:, :, 1:]               # if we discard the alpha chanel
                    writer.append_data(frame)
        self.pipe.send("{} saved video under {}".format(self.name, path))

    def rewards_to_return(self, rewards, prev_return=0):
        returns = np.zeros_like(rewards)
        for i in range(rewards.shape[-1] - 1, -1, -1):
            r = rewards[:, i]
            prev_return = r + self.discount_factor * prev_return
            returns[:, i] = prev_return
        return returns

    def update_reinforcement_learning(self, trajectories):
        states = trajectories["state_position"], trajectories["state_speed"], trajectories["state_tactile"]
        actions = trajectories["actions"]
        rewards = trajectories["rewards"]
        goals = trajectories["goals"]
        feed_dict = self.to_feed_dict(states=states, actions=actions, rewards=rewards, goals=goals)
        fetches = [
            self.global_step,
            self.train_op,
            self.summary]
        global_step, _, summary = self.sess.run(fetches, feed_dict=feed_dict)
        if global_step % 100 == 0:
            self.summary_writer.add_summary(summary, global_step=global_step)
            print("{} finished update number {}".format(self.name, global_step))
        return global_step

    def define_networks(self):
        n_trajectories = None
        trajectory_length = None
        self.inp_discrete_positions = tf.placeholder(shape=(n_trajectories, trajectory_length, ) + self._p_size, dtype=tf.float32)
        self.inp_discrete_speeds = tf.placeholder(shape=(n_trajectories, trajectory_length, ) + self._s_size, dtype=tf.float32)
        self.inp_tactile = tf.placeholder(shape=(n_trajectories, trajectory_length, ) + self._t_size, dtype=tf.float32)
        self.inp_goal = tf.placeholder(shape=(n_trajectories, trajectory_length) + self._t_size, dtype=tf.float32)
        self.actions_indices_placeholder = tf.placeholder(shape=(n_trajectories, trajectory_length, 4), dtype=tf.int32)
        ### can potentially be replaced with pure tensorflow: (calling python, like in asynchronous-aec)
        ### see the use of the scan function: https://github.com/tensorflow/agents/blob/master/tf_agents/utils/value_ops.py
        self.returns_not_bootstraped = tf.placeholder(shape=(n_trajectories, trajectory_length), dtype=tf.float32, name="returns_target")
        shape = tf.shape(self.inp_discrete_positions)
        n_trajectories = shape[0]
        trajectory_length = shape[1]
        increasing_discounted_gammas = tf.cumprod(tf.fill(dims=[trajectory_length], value=self.discount_factor), reverse=True)
        increasing_discounted_gammas = increasing_discounted_gammas[tf.newaxis, :, tf.newaxis]


        position_size = np.prod(self._p_size)
        speed_size = np.prod(self._s_size)
        tactile_size = np.prod(self._t_size)
        state_size = position_size + speed_size + tactile_size
        goal_size = tactile_size
        input_shape = (state_size + goal_size,)
        param_shape = (goal_size, )

        self.state = tf.concat(
            [tf.reshape(self.inp_discrete_positions, (n_trajectories, trajectory_length, position_size)),
             tf.reshape(self.inp_discrete_speeds, (n_trajectories, trajectory_length, speed_size)),
             self.inp_tactile],
            axis=-1)
        self.rl_inputs = tf.concat([self.state, self.inp_goal], axis=-1)
        parametrization_data = self.inp_goal

        ### DEFINE KERAS MODEL ###
        if self.parametrization_type.lower() == "none":
            ModelClass = RegularDenseNet
        elif self.parametrization_type.lower() == "concat":
            ModelClass = ConcatDenseNet
        elif self.parametrization_type.lower() == "affine":
            ModelClass = AffineDenseNet
        else:
            raise NotImplementedError("parametrization_type not recognized")

        n_joints = 4

        networks = [ModelClass(n_out=self.n_actions) for i in range(n_joints)]
        all_values = [network((self.state, parametrization_data)) for network in networks]
        all_values = tf.stack(all_values, axis=2)  # should have shape (N_TRAJECTORIES, TRAJECTORY_LENGTH, 4, N_ACTIONS)

        ntraj = tf.range(n_trajectories)
        tragl = tf.range(trajectory_length)
        jointn = tf.range(n_joints)
        grid_indices = tf.meshgrid(ntraj, tragl, jointn, indexing="ij")  # all have shape (N_TRAJECTORIES, TRAJECTORY_LENGTH, 4)
        indices = tf.stack(grid_indices + [self.actions_indices_placeholder], axis=-1) # should have shape (N_TRAJECTORIES, TRAJECTORY_LENGTH, 4, 4)
        values_of_picked_actions = tf.gather_nd(all_values, indices)  # should have shape (N_TRAJECTORIES, TRAJECTORY_LENGTH, 4)

        last_values = all_values[:, -1:]  # should have shape (N_TRAJECTORIES, 1, 4, N_ACTIONS)
        max_last_values = tf.reduce_max(last_values, axis=-1)  # shape should be (N_TRAJECTORIES, 1, 4)
        # shape should be (N_TRAJECTORIES, TRAJECTORY_LENGTH, 4):
        target_values =  self.returns_not_bootstraped[:, :, tf.newaxis] + increasing_discounted_gammas * max_last_values
        loss = tf.reduce_sum(tf.reduce_mean((target_values - values_of_picked_actions) ** 2, axis=1))
        mean_abs_distance = tf.reduce_mean(tf.abs(target_values - values_of_picked_actions))

        self.epsilon = tf.Variable(self.epsilon_init)
        self.greedy_actions_indices = tf.cast(tf.argmax(all_values, axis=-1), tf.int32)  # shape should be (N_TRAJECTORIES, TRAJECTORY_LENGTH, 4)
        random = tf.random_uniform(shape=(n_trajectories, trajectory_length, 4))
        random_indices = tf.random_uniform(shape=(n_trajectories, trajectory_length, 4), maxval=self.n_actions, dtype=tf.int32)
        condition = tf.greater(random, self.epsilon)
        self.sampled_actions_indices = tf.where(condition, self.greedy_actions_indices, random_indices)
        indices = tf.stack(grid_indices + [self.greedy_actions_indices], axis=-1) # should have shape (N_TRAJECTORIES, TRAJECTORY_LENGTH, 4, 4)
        self.greedy_q_value = tf.reduce_mean(tf.gather_nd(all_values, indices))
        indices = tf.stack(grid_indices + [self.sampled_actions_indices], axis=-1) # should have shape (N_TRAJECTORIES, TRAJECTORY_LENGTH, 4, 4)
        self.sampled_q_value = tf.reduce_mean(tf.gather_nd(all_values, indices))

        ### TRAIN OPS ###
        self.epsilon_update = self.epsilon.assign(self.epsilon * self.epsilon_decay)
        self.global_step = tf.Variable(0, dtype=tf.int32)
        self.global_step_inc = self.global_step.assign_add(1)
        optimizer = tf.train.AdamOptimizer(self.critic_lr)
        self.minimize_op = optimizer.minimize(loss)
        self.train_op = tf.group([self.epsilon_update, self.global_step_inc, self.minimize_op])

        ### SUMMARIES ###
        summary_loss = tf.sqrt(loss / tf.cast(n_trajectories * n_joints, tf.float32))
        loss_summary = tf.summary.scalar("/loss", summary_loss)
        raw_loss_summary = tf.summary.scalar("/raw_loss", loss)
        log_loss_summary = tf.summary.scalar("/log_loss", tf.log(summary_loss))
        mean_abs_distance_summary = tf.summary.scalar("/mean_abs_distance", mean_abs_distance)
        mean_return_summary = tf.summary.scalar("/mean_return", tf.reduce_mean(self.returns_not_bootstraped))
        self.summary = tf.summary.merge([
            loss_summary,
            log_loss_summary,
            raw_loss_summary,
            mean_abs_distance_summary,
            mean_return_summary])

    def get_state(self, as_multiple_trajectories=False, as_one_trajectory=False):
        if as_multiple_trajectories:
            return self.env.discrete_positions[np.newaxis, np.newaxis], self.env.discrete_speeds[np.newaxis, np.newaxis], self.env.tactile[np.newaxis, np.newaxis]
        if as_one_trajectory:
            return self.env.discrete_positions[np.newaxis], self.env.discrete_speeds[np.newaxis], self.env.tactile[np.newaxis]
        return self.env.discrete_positions, self.env.discrete_speeds, self.env.tactile

    def to_feed_dict(self, states=None, actions=None, rewards=None, goals=None):
        # transforms the inputs into a feed dict for the actor / critic
        feed_dict = {}
        if states is not None:
            feed_dict[self.inp_discrete_positions], feed_dict[self.inp_discrete_speeds], feed_dict[self.inp_tactile] = \
                states
        if actions is not None:
            feed_dict[self.actions_indices_placeholder] = actions
        if rewards is not None:
            # reverse pass through the rewards here...
            returns = self.rewards_to_return(rewards)
            feed_dict[self.returns_not_bootstraped] = returns
        if goals is not None:
            feed_dict[self.inp_goal] = goals
        return feed_dict


class Experiment:
    def __init__(self, n_parameter_servers, n_workers, experiment_dir, conf, display_dpi=3):
        self.n_parameter_servers = n_parameter_servers
        self.n_workers = n_workers
        self.experiment_dir = experiment_dir
        self.conf = conf
        self.mktree()
        # store infos related to the experiment
        with open(self.confdir + "/conf.txt", "w") as f:
            conf.dump(f)
        with open(self.confdir + "/conf.pkl", "wb") as f:
            pickle.dump(conf, f)
        with open(self.confdir + "/command_line.txt", "w") as f:
            f.write("python3 " + " ".join(sys.argv) + "\n")
        self.cluster = get_cluster(n_parameter_servers, n_workers)
        env = environment.Environment.from_conf(self.conf.environment_conf)
        goal_size = env.tactile.shape[0]
        vision_shape = env.vision.shape
        self.library_size = self.conf.goal_library_conf.goal_library_size
        ema_speed = self.conf.goal_library_conf.ema_speed
        self.goal_library = GoalLibrary(self.library_size, goal_size, vision_shape, ema_speed)
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
        print("EXPERIMENT: all processes started. Waiting for answer...")
        for p in self.here_pipes:
            print(p.recv())

    def mktree(self):
        self.logdir = self.experiment_dir + "/log"
        self.confdir = self.experiment_dir + "/conf"
        self.checkpointsdir = self.experiment_dir + "/checkpoints"
        self.videodir = self.experiment_dir + "/video"
        self.goaldir = self.experiment_dir + "/goals"
        self.goaldumpsdir = self.goaldir + "/dumps"
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(self.confdir, exist_ok=True)
        os.makedirs(self.videodir, exist_ok=True)
        os.makedirs(self.checkpointsdir, exist_ok=True)
        os.makedirs(self.goaldir, exist_ok=True)
        os.makedirs(self.goaldumpsdir, exist_ok=True)

    def parameter_server_func(self, task_index):
        server = tf.train.Server(self.cluster, "ps", task_index)
        server.join()

    def worker_func(self, task_index):
        worker = Worker(task_index, self.there_pipes[task_index], self.cluster, self.logdir, self.goal_library, self.conf)
        if task_index == 0:
            worker.initialize()
        else:
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

    def start_display_worker(self, training=True):
        self.here_display_pipes.append(self.here_worker_pipes[-1])
        self.here_worker_pipes = self.here_worker_pipes[:-1]
        self.here_display_pipes[-1].send(("run_display", training))

    def set_display_worker_idle(self):
        self.here_display_pipes[-1].send("done")  # quit run_display
        self.here_display_pipes[-1].recv()  # waiting
        self.here_worker_pipes.append(self.here_display_pipes[-1])
        self.here_display_pipes = self.here_display_pipes[:-1]

    def set_all_display_workers_idle(self):
        while len(self.here_display_pipes) > 0:
            self.set_display_worker_idle()

    def print_goal_library(self):
        self.here_worker_pipes[0].send(("print_goal_library", ))
        self.here_worker_pipes[0].recv()

    # def populate_goal_library(self):
    #     self.here_worker_pipes[0].send(("populate_goal_library", ))
    #     self.here_worker_pipes[0].recv()
    #     self.dump_goal_library()
    #     self.restore_goal_library(self.goaldumpsdir, all_same=True)

    def take_goal_library_snapshot(self):
        self.here_worker_pipes[0].send(("take_goal_library_snapshot", ))
        print(self.here_worker_pipes[0].recv())

    def restore_goal_library(self, path):
        for p in self.here_worker_pipes:
            p.send(("restore_goal_library", path))
        for p in self.here_worker_pipes:
            p.recv()

    def dump_goal_library(self):
        self.here_worker_pipes[0].send(("dump_goal_library", self.goaldumpsdir))
        print(self.here_worker_pipes[0].recv())

    def randomize_env(self, n=None):
        for p in self.here_worker_pipes:
            p.send(("randomize_env", n, True))
        for p in self.here_worker_pipes:
            p.recv()

    def save_vision_related_to_goals(self):
        for p in self.here_worker_pipes:
            p.send(("save_vision_related_to_goals", self.goaldir))
        for p in self.here_worker_pipes:
            p.recv()

    def asynchronously_train(self, n_updates, train_actor=True):
        for p in self.here_worker_pipes:
            p.send(("run_training", n_updates, train_actor))
        for p in self.here_worker_pipes:
            p.recv()

    def save_model(self):
        self.here_worker_pipes[0].send(("save", self.checkpointsdir))
        print(self.here_worker_pipes[0].recv())

    def save_video(self, name, path=None, n_frames=2000, training=False):
        path = self.videodir + "/" + name + "/" if path is None else path + "/" + name + "/"
        os.mkdir(path)
        n_workers = len(self.here_worker_pipes)
        for i in range(self.library_size):
            self.here_worker_pipes[i % n_workers].send(("save_video", path, i, n_frames, training))
        for i in range(self.library_size):
            print(self.here_worker_pipes[i % n_workers].recv())

    def save_video_all_goals(self, name, path=None, n_frames=50, training=False):
        path = self.videodir + "/" + name + "/" if path is None else path + "/" + name + "/"
        os.mkdir(path)
        n_workers = len(self.here_worker_pipes)
        self.here_worker_pipes[0].send(("save_video_all_goals", path, n_frames, training))
        print(self.here_worker_pipes[0].recv())

    def save_contact_logs(self, name):
        for p in self.here_worker_pipes:
            p.send(("save_contact_logs", name))
        for p in self.here_worker_pipes:
            p.recv()

    def restore_model(self, path):
        self.here_worker_pipes[0].send(("restore", path))
        print(self.here_worker_pipes[0].recv())

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
