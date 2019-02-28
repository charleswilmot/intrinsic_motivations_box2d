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
    def __init__(self, task_index, pipe, env, cluster, logdir, discount_factor, sequence_length, reward_params,
                 model_lr, critic_lr, actor_lr):
        self.task_index = task_index
        self.cluster = cluster
        self._n_workers = self.cluster.num_tasks("worker") - 1
        self.job_name = "worker"
        self.server = tf.train.Server(cluster, self.job_name, task_index)
        self.name = "/job:{}/task:{}".format(self.job_name, task_index)
        self.device = tf.train.replica_device_setter(worker_device=self.name, cluster=cluster)
        self.env = env
        self.discount_factor = discount_factor
        self.sequence_length = sequence_length
        self.reward_params = reward_params
        self.model_lr = model_lr
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.pipe = pipe
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

    def run_reinforcement_learning(self, n_updates, train_actor=True):
        global_rl_step = self.sess.run(self.global_rl_step)
        n_updates += global_rl_step
        while global_rl_step < n_updates - self._n_workers:
            # Collect some experience
            transitions = self.run_n_rl_steps()
            # Update the global networks
            global_rl_step = self.update_reinforcement_learning(*transitions, train_actor=train_actor)
        self.summary_writer.flush()
        self.pipe.send("{} going IDLE".format(self.name))

    def run_all(self, n_updates, train_actor=True):
        global_rl_step = self.sess.run(self.global_rl_step)
        n_updates += global_rl_step
        while global_rl_step < n_updates - self._n_workers:
            # Collect some experience
            transitions = self.run_n_rl_steps()
            # Update the global networks
            global_rl_step = self.update_all(*transitions, train_actor=train_actor)
        self.summary_writer.flush()
        self.pipe.send("{} going IDLE".format(self.name))

    def run_display(self, sample=True):
        win = viewer.JointAgentWindow(self.discount_factor, return_lookback=50)
        while not self.pipe.poll():
            self.run_n_display_steps(win, sample)
        win.close()
        self.pipe.recv()  # done
        self.pipe.send("{} (display) going IDLE".format(self.name))

    def run_model(self, n_updates):
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
        action = self.sess.run(self.stochastic_actions, feed_dict=feed_dict)
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
        action_return_fetches = [self.stochastic_actions if sample else self.greedy_actions, self.critic_value]
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
        rewards, model_summary = self.sess.run([self.rewards, self.model_summary], feed_dict=feed_dict)
        feed_dict = self.to_rl_feed_dict(states=rl_states, actions=actions, rewards=rewards)
        train_op = self.rl_train_op if train_actor else self.critic_train_op
        fetches = [self.actor_loss, self.critic_loss, self.global_rl_step_inc, train_op, self.rl_summary]
        aloss, closs, global_rl_step, _, rl_summary = self.sess.run(fetches, feed_dict=feed_dict)
        self.summary_writer.add_summary(model_summary, global_step=global_rl_step)
        self.summary_writer.add_summary(rl_summary, global_step=global_rl_step)
        if global_rl_step % 100 <= self._n_workers:
            self.summary_writer.flush()
        print("{} finished update number {} (actor loss = {:.3f}  critic loss = {:.3f})".format(
              self.name, global_rl_step, aloss, closs))
        return global_rl_step

    def update_model(self, states):
        feed_dict = self.to_model_feed_dict(states=states, targets=True)
        fetches = [self.model_loss, self.global_model_step_inc, self.model_train_op, self.model_summary]
        loss, global_model_step, _, model_summary = self.sess.run(fetches, feed_dict=feed_dict)
        self.summary_writer.add_summary(model_summary, global_step=global_model_step)
        if global_model_step % 100 <= self._n_workers:
            self.summary_writer.flush()
        print("{} finished update number {} (loss = {:.3f})".format(self.name, global_model_step, loss))
        return global_model_step

    def update_all(self, model_states, rl_states, actions, train_actor=True):
        feed_dict = self.to_model_feed_dict(states=model_states, targets=True)
        fetches = [self.rewards, self.model_loss, self.model_train_op, self.model_summary, self.global_rl_step]
        rewards, mloss, _, model_summary, global_rl_step = self.sess.run(fetches, feed_dict=feed_dict)
        self.summary_writer.add_summary(model_summary, global_step=global_rl_step)
        feed_dict = self.to_rl_feed_dict(states=rl_states, actions=actions, rewards=rewards)
        train_op = self.rl_train_op if train_actor else self.critic_train_op
        fetches = [self.actor_loss, self.critic_loss, self.global_rl_step_inc, train_op, self.rl_summary]
        aloss, closs, global_rl_step, _, rl_summary = self.sess.run(fetches, feed_dict=feed_dict)
        self.summary_writer.add_summary(rl_summary, global_step=global_rl_step)
        if global_rl_step % 100 <= self._n_workers:
            self.summary_writer.flush()
        print("{} finished update number {} (model loss = {:.3f}  critic loss = {:.3f})".format(
              self.name, global_rl_step, mloss, closs))
        return global_rl_step


class JointAgentWorker(Worker):
    def define_networks(self):
        self.define_net_dims()
        with tf.device(self.device):
            # define model
            self.define_model()
            # define reinforcement learning
            self.define_reinforcement_learning()

    def define_net_dims(self):
        self.n_discrete = self.env._n_discrete
        self.model_net_dim = [self.n_discrete * 3, 600, 600, 600, self.n_discrete]
        self.critic_net_dim = [4 + self.n_discrete * 4 * 2, 600, 1]
        self.actor_net_dim = [self.n_discrete * 4 * 2, 600, 4]

    def define_model(self):
        #############################
        # MUST DEFINE : #############
        #############################
        # self.model_outputs
        # self.rewards
        # self.model_loss
        # self.model_losses
        # self.global_model_step
        # self.model_train_op
        # self.model_summary
        # self.model_summary_at_rl_time   --> missing in current ipl, TODO  --> useless ?
        # PLACEHOLDERS : ############
        # self.model_inputs
        # self.model_targets
        #############################
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
        self.define_reward(**self.reward_params)
        self.global_model_step = tf.Variable(0, dtype=tf.int32)
        self.global_model_step_inc = self.global_model_step.assign_add(1)
        # optimizer / summary
        self.model_optimizer = tf.train.AdamOptimizer(1e-3)
        self.model_optimizer = tf.train.AdamOptimizer(self.model_lr)
        self.model_train_op = self.model_optimizer.minimize(self.model_loss)
        sum_model_loss = tf.summary.scalar("/model/loss", self.model_losses[0])
        sum_model_reward = tf.summary.scalar("/model/reward", self.rewards[0])
        self.model_summary = tf.summary.merge([sum_model_loss, sum_model_reward])

    def define_reinforcement_learning(self):
        #############################
        # MUST DEFINE : #############
        #############################
        # self.stochastic_actions
        # self.greedy_actions
        # self.critic_value
        # self.critic_loss
        # self.actor_loss
        # self.rl_train_op
        # self.critic_train_op
        # self.global_rl_step
        # self.global_both_step  --> must be automalicaly incremented when calling some train op...
        # self.rl_summary
        # PLACEHOLDERS : ############
        # self.rl_inputs
        # self.actions
        # self.return_targets_not_bootstraped
        #############################
        self.rl_inputs = tf.placeholder(shape=[None, self.actor_net_dim[0]], dtype=tf.float32, name="actor_inputs")
        self.return_targets_not_bootstraped = tf.placeholder(shape=[None], dtype=tf.float32, name="returns_target")
        self.actions = tf.placeholder(shape=[None, self.actor_net_dim[-1]], dtype=tf.float32, name="actions")
        with tf.variable_scope("actor_net"):
            prev_layer = self.rl_inputs
            for i, d in enumerate(self.actor_net_dim[1:]):
                activation_fn = tf.nn.relu if i < len(self.actor_net_dim) - 2 else None
                prev_layer = tl.fully_connected(prev_layer, d, scope="layer{}".format(i), activation_fn=activation_fn)
            self.greedy_actions = prev_layer
            self.stochastic_actions = self.greedy_actions + tf.random_normal(shape=tf.shape(self.greedy_actions), stddev=0.05)
            actor_vars = [x for x in tf.global_variables() if x.name.startswith("actor_net")]
        with tf.variable_scope("critic_net"):
            prev_layer = tf.concat([self.rl_inputs, self.actions], axis=1)
            for i, d in enumerate(self.critic_net_dim[1:]):
                activation_fn = tf.nn.relu if i < len(self.critic_net_dim) - 2 else None
                prev_layer = tl.fully_connected(prev_layer, d, scope="layer{}".format(i), activation_fn=activation_fn)
            self.critic_value = tf.squeeze(prev_layer, axis=1, name="critic_value")
        with tf.variable_scope("critic_net"):
            prev_layer = tf.concat([self.rl_inputs, self.greedy_actions], axis=1)  # could also add noise (see stochastic)
            for i, d in enumerate(self.critic_net_dim[1:]):
                activation_fn = tf.nn.relu if i < len(self.critic_net_dim) - 2 else None
                prev_layer = tl.fully_connected(prev_layer, d, scope="layer{}".format(i), activation_fn=activation_fn, reuse=True)
            actor_value = tf.squeeze(prev_layer, axis=1, name="critic_value")
        # losses
        constant_gammas = tf.fill(dims=[tf.shape(self.rl_inputs)[0]], value=self.discount_factor)
        increasing_discounted_gammas = tf.cumprod(constant_gammas, reverse=True)
        return_targets = self.return_targets_not_bootstraped + increasing_discounted_gammas * self.critic_value[-1]
        self.critic_losses = (return_targets - self.critic_value) ** 2  # * (1 - increasing_discounted_gammas)
        self.critic_loss = tf.reduce_mean(self.critic_losses, name="critic_loss")
        self.actor_loss = -tf.reduce_mean(actor_value)
        # train ops
        self.global_rl_step = tf.Variable(0, dtype=tf.int32)
        self.global_rl_step_inc = self.global_rl_step.assign_add(1)
        with tf.control_dependencies([self.global_rl_step_inc]):
            self.critic_train_op = tf.train.AdamOptimizer(1e-3).minimize(self.critic_loss)
            self.actor_train_op = tf.train.AdamOptimizer(1e-3).minimize(self.actor_loss, var_list=actor_vars)
            self.critic_train_op = tf.train.AdamOptimizer(self.critic_lr).minimize(self.critic_loss)
            self.actor_train_op = tf.train.AdamOptimizer(self.actor_lr).minimize(self.actor_loss, var_list=actor_vars)
        with tf.control_dependencies([self.critic_train_op]):
            with tf.control_dependencies([self.actor_train_op]):
                self.rl_train_op = tf.no_op()
        # summaries
        sum_actor_loss = tf.summary.scalar("/rl/actor_loss", -actor_value[0])
        sum_critic_loss = tf.summary.scalar("/rl/critic_loss", self.critic_losses[0])
        critic_quality = self.critic_losses[0] / tf.reduce_mean((return_targets - tf.reduce_mean(return_targets)) ** 2)
        sum_critic_quality = tf.summary.scalar("/rl/critic_quality", critic_quality)
        sum_policy_gradient = tf.summary.scalar("/rl/policy_grad", tf.reduce_mean(tf.abs(tf.gradients(actor_value, self.greedy_actions))))
        self.rl_summary = tf.summary.merge([sum_actor_loss, sum_critic_loss, sum_critic_quality, sum_policy_gradient])

    def get_model_state(self):
        return self.env.discrete_positions, self.env.discrete_speeds, self.env.discrete_target_positions

    def get_rl_state(self):
        return self.env.discrete_positions, self.env.discrete_speeds

    def to_rl_feed_dict(self, states=None, actions=None, rewards=None):
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
            feed_dict[self.return_targets_not_bootstraped] = returns
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


class MinimizeJointAgentWorker(JointAgentWorker):
    def define_reward(self, model_loss_converges_to):
        reward_scale = (1 - self.discount_factor) / model_loss_converges_to
        self.rewards = - reward_scale * self.model_losses


class MaximizeJointAgentWorker(JointAgentWorker):
    def define_reward(self, model_loss_converges_to):
        reward_scale = (1 - self.discount_factor) / model_loss_converges_to
        self.rewards = reward_scale * self.model_losses


class TargetErrorJointAgentWorker(JointAgentWorker):
    def define_reward(self, target_prediction_error):
        self.rewards = -tf.abs(self.model_losses - target_prediction_error) * (1 - self.discount_factor) / 0.04


class RangeErrorJointAgentWorker(JointAgentWorker):
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

    def asynchronously_run_model(self, n_updates):
        for p in self.here_worker_pipes:
            p.send(("run_model", n_updates))
        for p in self.here_worker_pipes:
            p.recv()

    def asynchronously_run_reinforcement_learning(self, n_updates, train_actor=True):
        for p in self.here_worker_pipes:
            p.send(("run_reinforcement_learning", n_updates, train_actor))
        for p in self.here_worker_pipes:
            p.recv()

    def asynchronously_run_both(self, n_updates, train_actor=True):
        for p in self.here_worker_pipes:
            p.send(("run_all", n_updates, train_actor))
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
