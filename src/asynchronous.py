import sys
import pickle
from discretization import np_discretization_reward
from scipy import special
from tempfile import TemporaryDirectory
import png
import environment
from replay_buffer import Buffer
from dtgoal2 import GoalWarehouse
import discretization
import socket
import multiprocessing
import subprocess
import numpy as np
from numpy import pi, log
import tensorflow as tf
import tensorflow.contrib.layers as tl
import time
import viewer
import os
import filelock



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
    def __init__(self, task_index, pipe, env, cluster, logdir, discount_factor, sequence_length,
                 critic_lr, actor_lr, entropy_coef, softmax_temperature, replay_buffer_size, updates_per_episode,
                 HER_strategy, goals_warehouse_ema_speed):
        self.task_index = task_index
        self.cluster = cluster
        self._n_workers = self.cluster.num_tasks("worker") - 1
        self.job_name = "worker"
        self.server = tf.train.Server(cluster, self.job_name, task_index)
        self.name = "/job:{}/task:{}".format(self.job_name, task_index)
        self.device = tf.train.replica_device_setter(worker_device=self.name, cluster=cluster)
        self.env = env
        self.logdir = logdir
        self.discount_factor = discount_factor
        self.sequence_length = sequence_length
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.entropy_coef = entropy_coef
        self.softmax_temperature = softmax_temperature
        self.replay_buffer_size = replay_buffer_size
        self.updates_per_episode = updates_per_episode
        self.HER_strategy = HER_strategy
        self.n_actions = 30
        self.dtgoal = np.dtype(
            [("intensities", np.float32, 5),
             ("all", np.float32),
             ("pursued", np.float32),
             ("not_pursued", np.float32)])
        max_goals = 1000
        self.n_goals = 0
        self.actions_ground_to_arm = np.linspace(-3.14, 3.14, self.n_actions)
        self.actions_arm_to_arm = np.linspace(-2.7, 2.7, self.n_actions)
        self.pipe = pipe
        self.replay_buffer = Buffer(self.replay_buffer_size)
        p, s, t = self.get_state()
        self._p_size, self._s_size, self._t_size = p.shape, s.shape, t.shape
        self.goal_temperature = 0.015
        self.goals_warehouse_ema_speed = goals_warehouse_ema_speed
        self.goals_warehouse = GoalWarehouse(100, self._t_size[0], goals_warehouse_ema_speed)
        self.define_networks()
        # graph = tf.get_default_graph() if task_index == 0 else None
        graph = None
        self.summary_writer = tf.summary.FileWriter(self.logdir + "/worker{}".format(task_index), graph=graph)
        self.saver = tf.train.Saver()
        self.sess = tf.Session(target=self.server.target)
        if task_index == 0 and len(self.sess.run(tf.report_uninitialized_variables())) > 0:  # todo: can be done in Experiment
            self.sess.run(tf.global_variables_initializer())
            print("{}  variables initialized".format(self.name))


    def actions_dict_from_indices(self, actions):
        return {
            "Arm1_to_Arm2_Left": self.actions_arm_to_arm[actions[0]],
            "Ground_to_Arm1_Left": self.actions_ground_to_arm[actions[1]],
            "Arm1_to_Arm2_Right": self.actions_arm_to_arm[actions[2]],
            "Ground_to_Arm1_Right": self.actions_ground_to_arm[actions[3]]
        }

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
            # returns a trajectory return (state_positions, state_speeds, state_tactile), goals, actions, rewards
            states, goals, actions, rewards = self.run_n_steps()
            # Place in buffer
            if self.HER_strategy.lower() == "none":
                self.replay_buffer.incorporate((states, goals, actions, rewards))
            elif self.HER_strategy.lower() == "first_contact":
                for tactile in states[2]:
                    if np.sum(tactile) > 0:
                        faked_goals = np.repeat(tactile, self.sequence_length, 0)
                        faked_rewards = np_discretization_reward(states[2], faked_goals)
                        self.replay_buffer.incorporate((states, faked_goals, actions, faked_rewards))
                        break
            elif self.HER_strategy.lower() == "all_contacts":
                for tactile in states[2]:
                    if np.sum(tactile) > 0:
                        faked_goals = np.repeat(tactile[np.newaxis, :], self.sequence_length, 0)
                        faked_rewards = np_discretization_reward(states[2], faked_goals)
                        self.replay_buffer.incorporate((states, faked_goals, actions, faked_rewards))
            elif self.HER_strategy.lower() == "something_else":
                raise NotImplementedError("not implemented")
                pass
            # Update the global networks
            trajectories = self.replay_buffer.batch(self.updates_per_episode)
            for trajectory in trajectories:
                global_step = self.update_reinforcement_learning(*trajectory, train_actor=train_actor)
                if global_step >= n_updates - self._n_workers:
                    break
        self.summary_writer.flush()
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

    def print_goal_warehouse(self):
        self.goals_warehouse.print_with_header("{} goal warehouse:".format(self.name))
        self.pipe.send("{} printed goal warehouse".format(self.name))

    def run_video(self, path, n_sequences, training=True):
        start_index = 0
        with TemporaryDirectory() as tmppath:
            while not os.path.isdir(tmppath):
                time.sleep(0.1)
            for i in range(n_sequences):
                start_index = self.run_n_video_steps(tmppath, start_index, training=training)
            os.system("ffmpeg -loglevel panic -r 24 -i {}/frame_%06d.png -vcodec mpeg4 -b 100000 -y {}".format(tmppath, path))
        self.pipe.send("{} saved video under {}".format(self.name, path))

    def take_goals_warehouse_snapshot(self):
        iteration = self.sess.run(self.global_step)
        self.goals_warehouse.take_snapshot(iteration)
        self.pipe.send("{} took a snapshot of the goal warehouse".format(self.name))

    def restore_goals_warehouse(self, path):
        with open(path + "/worker_{}.pkl".format(self.task_index), "rb") as f:
            self.goals_warehouse = pickle.load(f)
        self.pipe.send("{} restored the goal warehouse".format(self.name))

    def dump_goals_warehouse(self, path):
        self.goals_warehouse.dump(path + "/worker_{}.pkl".format(self.task_index))
        self.pipe.send("{} saved warehouse under {}".format(self.name, path))

    def save_contact_logs(self, name):
        path = self.logdir + "/worker{}/contacts_{}.pkl".format(self.task_index, name)
        self.env.save_contact_logs(path)
        self.pipe.send("{} saved contact logs under {}".format(self.name, path))

    def get_action(self, goals):
        feed_dict = self.to_feed_dict(states=self.get_state(True), goals=goals)
        action = self.sess.run(self.action_applied_in_env, feed_dict=feed_dict)
        return action[0]

    def run_n_steps(self):
        state_positions = np.zeros((self.sequence_length, ) + self._p_size)
        state_speeds = np.zeros((self.sequence_length, ) + self._s_size)
        state_tactile = np.zeros((self.sequence_length, ) + self._t_size)
        goal = self.goals_warehouse.select_goal_learning_potential(self.goal_temperature)
        goals = np.repeat(goal[np.newaxis, :], self.sequence_length, 0)
        actions = np.zeros((self.sequence_length, 4))
        rewards = np.zeros((self.sequence_length, ))
        for iteration in range(self.sequence_length):
            # get action
            action = self.get_action(goals[:1])
            actions[iteration] = action
            # set action
            self.env.set_positions(self.actions_dict_from_indices(action))
            # get states
            state_positions[iteration], state_speeds[iteration], state_tactile[iteration] = self.get_state()
            reward = self.goals_warehouse.add_goal(
                state_tactile[iteration],
                current_goal=goal,
                vision=self.env.vision)  # point is to have a snapshot of the contact stored in the warehouse
            rewards[iteration] = reward  # different way of computing the reward (numpy vs tensorflow) see run_n_display_steps
            # run environment step
            self.env.env_step()
        self.randomize_env()
        return (state_positions, state_speeds, state_tactile), goals, actions, rewards

    def randomize_env(self):
        for i in range(2):
            action = np.random.randint(0, self.n_actions, 4)
            self.env.set_positions(self.actions_dict_from_indices(action))
            self.env.env_step()

    def run_n_display_steps(self, win, training=True):
        fetches = [self.action_applied_in_env if training else self.greedy_actions_indices, self.critic_value, self.rewards]
        rewards = np.zeros(self.sequence_length)
        goal = self.goals_warehouse.select_goal_learning_potential(self.goal_temperature)
        goals = goal[np.newaxis, :]
        for i in range(self.sequence_length):
            feed_dict = self.to_feed_dict(states=self.get_state(True), goals=goals)
            actions, predicted_returns, reward = self.sess.run(fetches, feed_dict=feed_dict)
            action = actions[0]
            rewards[i] = reward[0]
            # set positions in env
            self.env.set_positions(self.actions_dict_from_indices(action))
            # run action in env
            self.env.env_step()
            # get current vision
            self.goals_warehouse.add_goal(
                self.env.tactile,
                current_goal=goal,
                vision=self.env.vision)  # point is to have a snapshot of the contact stored in the warehouse
            vision = self.env.vision
            tactile_true = self.env.tactile
            tactile_target = goals[0]
            current_reward = rewards[i]
            predicted_return = predicted_returns[0]
            # display
            win(vision, tactile_true, tactile_target, current_reward, predicted_return)

    def save_vision_related_to_goals(self, goaldir):
        # func : i, data -> action
        self.goals_warehouse.save_vision(goaldir + "/worker{}/".format(self.task_index))
        self.pipe.send("{} saved vision related to the goals in the goals warehouse".format(self.name))

    def run_n_video_steps(self, path, start_index, training=True):
        action_fetches = self.action_applied_in_env if training else self.greedy_actions
        goals = self.goals_warehouse.select_goal_learning_potential(self.goal_temperature)[np.newaxis, :]
        for i in range(self.sequence_length):
            feed_dict = self.to_feed_dict(states=self.get_state(True), goals=goals)
            actions = self.sess.run(action_fetches, feed_dict=feed_dict)
            action = actions[0]
            # set positions in env
            action_dict = self.actions_dict_from_indices(action)
            self.env.set_positions(action_dict)
            # run action in env
            self.env.env_step()
            # get current vision
            vision = self.env.vision
            # save
            data = vision.reshape(vision.shape[0], -1)
            png.from_array(data, "RGB").save(path + "/frame_{:06d}.png".format(start_index + i))
        return start_index + i + 1

    def rewards_to_return(self, rewards, prev_return=0):
        returns = np.zeros_like(rewards)
        for i in range(len(rewards) - 1, -1, -1):
            r = rewards[i]
            prev_return = r + self.discount_factor * prev_return
            returns[i] = prev_return
        return returns

    def update_reinforcement_learning(self, states, goals, actions, rewards, train_actor=True):
        ### Placeholders:
        # self.inp_discrete_positions
        # self.inp_discrete_speeds
        # self.inp_tactile
        # self.inp_goals
        # self.return_targets_not_bootstraped
        # self.actions_indices
        state_positions, state_speeds, state_tactile = states
        feed_dict = self.to_feed_dict(states=states, goals=goals, actions=actions, rewards=rewards)
        if train_actor:
            fetches = [self.global_step_inc, self.critic_train_op, self.actor_train_op, self.rl_summary]
            global_step, _, _, summary = self.sess.run(fetches, feed_dict=feed_dict)
        else:
            fetches = [self.global_step_inc, self.critic_train_op, self.rl_summary]
            global_step, _, summary = self.sess.run(fetches, feed_dict=feed_dict)
        self.summary_writer.add_summary(summary, global_step=global_step)
        if global_step % 100 == 0:
            print("{} finished update number {}".format(self.name, global_step))
        return global_step

    def define_networks(self):
        #############################
        # MUST DEFINE : #############
        #############################
        # self.sampled_actions
        # self.greedy_actions
        # self.critic_value
        # self.critic_loss
        # self.actor_loss
        # self.rl_train_op
        # self.critic_train_op
        # self.global_step
        # self.rl_summary
        # PLACEHOLDERS : ############
        # self.rl_inputs
        # self.actions
        # self.return_targets_not_bootstraped
        #############################
        critic_net_dim = [None, 200, 200, 1]
        self.inp_discrete_positions = tf.placeholder(shape=(None, ) + self._p_size, dtype=tf.float32)
        self.inp_discrete_speeds = tf.placeholder(shape=(None, ) + self._s_size, dtype=tf.float32)
        self.inp_tactile = tf.placeholder(shape=(None, ) + self._t_size, dtype=tf.float32)
        self.inp_goals = tf.placeholder(shape=(None, ) + self._t_size, dtype=tf.float32)
        self.rewards = discretization.tf_discretization_reward(self.inp_goals, self.inp_tactile)
        self.rl_inputs = tf.concat(
            [tf.reshape(self.inp_discrete_positions, (-1, np.prod(self._p_size))),
             tf.reshape(self.inp_discrete_speeds, (-1, np.prod(self._s_size))),
             self.inp_tactile,
             self.inp_goals],
            axis=-1)
        self.return_targets_not_bootstraped = tf.placeholder(shape=[None], dtype=tf.float32, name="returns_target")
        batch_size = tf.shape(self.rl_inputs)[0]
        batch_size_64 = tf.cast(batch_size, tf.int64)
        constant_gammas = tf.fill(dims=[batch_size], value=self.discount_factor)
        increasing_discounted_gammas = tf.cumprod(constant_gammas, reverse=True)
        prev_layer = self.rl_inputs
        with tf.variable_scope("critic"):
            for i, d in enumerate(critic_net_dim[1:]):
                prev_layer = tl.fully_connected(prev_layer, d, scope="layer{}".format(i), activation_fn=lrelu)
            self.critic_value = tf.squeeze(prev_layer, axis=1, name="critic_value")
        self.return_targets = self.return_targets_not_bootstraped + increasing_discounted_gammas * tf.stop_gradient(self.critic_value[-1])
        self.critic_losses = (self.critic_value - self.return_targets) ** 2
        self.critic_loss = tf.reduce_mean(self.critic_losses, name="loss")

        self.actions_indices = tf.placeholder(shape=[None, 4], dtype=tf.int32, name="actor_picked_actions")  # picked actions
        self.actor_targets = self.return_targets - tf.stop_gradient(self.critic_value)
        targets = tf.expand_dims(self.actor_targets, -1)
        actor_net_dim = [None, 200, 200, self.n_actions]
        actors_out = []
        with tf.variable_scope("actor"):
            # for every joint:
            for j in range(4):
                prev_layer = self.rl_inputs
                for i, d in enumerate(actor_net_dim[1:]):
                    prev_layer = tl.fully_connected(prev_layer, d, scope="actor{}_layer{}".format(j, i))
                actors_out.append(prev_layer)
            self.logits = tf.stack(actors_out, axis=1) / self.softmax_temperature  # [BS, 4, NA]
            self.probs = tf.nn.softmax(self.logits, axis=-1)
            # self.dist = tf.distributions.Categorical(logits=self.logits)
            self.dist = tf.distributions.Categorical(probs=self.probs)
            self.log_picked_probs = self.dist.log_prob(self.actions_indices)  # [BS, 4]
            self.picked_probs = self.dist.prob(self.actions_indices)  # [BS, 4]
            self.greedy_actions_indices = tf.argmax(self.logits, axis=-1)   # [BS, 4]
            self.sampled_actions_indices = self.dist.sample()

        self.action_applied_in_env = self.sampled_actions_indices
        self.entropy = self.dist.entropy()
        self.entropy_mean = tf.reduce_mean(self.entropy, name="entropy_mean")
        # self.actor_losses = -tf.maximum(self.log_picked_probs, -4) * targets - self.entropy_coef * self.entropy
        self.actor_losses = - self.log_picked_probs * targets - self.entropy_coef * self.entropy
        self.actor_loss = tf.reduce_sum(self.actor_losses, name="loss")
        # optimizer / summary
        self.global_step = tf.Variable(0, dtype=tf.int32)
        self.global_step_inc = self.global_step.assign_add(1)
        self.critic_optimizer = tf.train.AdamOptimizer(self.critic_lr)
        self.critic_train_op = self.critic_optimizer.minimize(self.critic_loss)
        self.actor_optimizer = tf.train.AdamOptimizer(self.actor_lr)
        self.actor_train_op = self.actor_optimizer.minimize(self.actor_loss)
        # summaries
        names = ["arm1_arm2_left", "arm1_arm2_right", "ground_arm1_left", "ground_arm1_right"]
        prob_summaries = []
        for i, name in zip(range(4), names):
            prob_summary = tf.summary.scalar("joints/{}_prob".format(name), self.picked_probs[0, i])
            prob_summaries.append(prob_summary)
        max_logits_summary = tf.summary.scalar("/rl/max_logits", tf.reduce_max(self.logits))
        entropy_summary = tf.summary.scalar("/rl/entropy", self.entropy_mean)
        actor_loss_summary = tf.summary.scalar("/rl/actor_loss", tf.reduce_mean(self.actor_losses))
        critic_loss_summary = tf.summary.scalar("/rl/critic_loss", self.critic_losses[0])
        critic_quality = tf.reduce_mean(self.critic_losses / tf.reduce_mean((self.return_targets - tf.reduce_mean(self.return_targets)) ** 2))
        sum_critic_quality = tf.summary.scalar("/rl/critic_quality", tf.clip_by_value(critic_quality, -20, 20))
        self.rl_summary = tf.summary.merge([
            max_logits_summary,
            entropy_summary,
            actor_loss_summary,
            critic_loss_summary,
            sum_critic_quality] + prob_summaries)

    def get_state(self, as_trajectory=False):
        if as_trajectory:
            return self.env.discrete_positions[np.newaxis, :], self.env.discrete_speeds[np.newaxis, :], self.env.tactile[np.newaxis, :]
        return self.env.discrete_positions, self.env.discrete_speeds, self.env.tactile

    def to_feed_dict(self, states=None, goals=None, actions=None, rewards=None):
        # transforms the inputs into a feed dict for the actor / critic
        feed_dict = {}
        if states is not None:
            feed_dict[self.inp_discrete_positions], feed_dict[self.inp_discrete_speeds], feed_dict[self.inp_tactile] = \
                states
        if goals is not None:
            feed_dict[self.inp_goals] = goals
        if actions is not None:
            feed_dict[self.actions_indices] = actions
        if rewards is not None:
            # reverse pass through the rewards here...
            returns = self.rewards_to_return(rewards)
            feed_dict[self.return_targets_not_bootstraped] = returns
        return feed_dict


class Experiment:
    def __init__(self, n_parameter_servers, n_workers, experiment_dir, args_env, args_worker, display_dpi=3):
        lock = filelock.FileLock("/home/wilmot/Documents/code/intrinsic_motivations_box2d/experiments/lock")
        lock.acquire()
        self.n_parameter_servers = n_parameter_servers
        self.n_workers = n_workers
        self.experiment_dir = experiment_dir
        self.mktree()
        # store infos related to the experiment
        with open(self.confdir + "/worker_conf.pkl", "wb") as f:
            pickle.dump(args_worker, f)
        with open(self.confdir + "/env_conf.pkl", "wb") as f:
            pickle.dump(args_env, f)
        with open(self.confdir + "/command_line.txt", "w") as f:
            f.write("python3 " + " ".join(sys.argv) + "\n")
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
        self.confdir = self.experiment_dir + "/conf"
        self.checkpointsdir = self.experiment_dir + "/checkpoints"
        self.videodir = self.experiment_dir + "/video"
        self.goaldir = self.experiment_dir + "/goals"
        self.goaldumpsdir = self.goaldir + "/dumps"
        os.mkdir(self.experiment_dir)
        os.mkdir(self.logdir)
        os.mkdir(self.confdir)
        os.mkdir(self.videodir)
        os.mkdir(self.checkpointsdir)
        os.mkdir(self.goaldir)
        os.mkdir(self.goaldumpsdir)

    def parameter_server_func(self, task_index):
        server = tf.train.Server(self.cluster, "ps", task_index)
        server.join()

    def worker_func(self, task_index):
        env = environment.Environment(*self.args_env)
        worker = Worker(task_index, self.there_pipes[task_index], env, *self.args_worker)
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

    def print_goal_warehouse(self):
        for p in self.here_worker_pipes:
            p.send(("print_goal_warehouse", ))
        for p in self.here_worker_pipes:
            p.recv()

    def take_goals_warehouse_snapshot(self):
        for p in self.here_worker_pipes:
            p.send(("take_goals_warehouse_snapshot", ))
        for p in self.here_worker_pipes:
            p.recv()

    def restore_goals_warehouse(self, path):
        for p in self.here_worker_pipes:
            p.send(("restore_goals_warehouse", path))
        for p in self.here_worker_pipes:
            p.recv()

    def dump_goals_warehouse(self):
        for p in self.here_worker_pipes:
            p.send(("dump_goals_warehouse", self.goaldumpsdir))
        for p in self.here_worker_pipes:
            p.recv()

    def randomize_env(self):
        for p in self.here_worker_pipes:
            p.send(("randomize_env", ))
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

    def save_video(self, name, n_sequences, training=True):
        path = self.videodir + "/{}.mp4".format(name)
        self.here_worker_pipes[0].send(("run_video", path, n_sequences, training))
        print(self.here_worker_pipes[0].recv())

    def save_contact_logs(self, name):
        for p in self.here_worker_pipes:
            p.send(("save_contact_logs", name))
        for p in self.here_worker_pipes:
            p.recv()

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
