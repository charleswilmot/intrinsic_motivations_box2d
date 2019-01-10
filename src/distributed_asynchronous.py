import multiprocessing
import numpy as np
from numpy import pi, log
from numpy.random import normal
from tempfile import TemporaryDirectory
import tensorflow as tf
import tensorflow.contrib.layers as tl
import time
import viewer
import os


def actions_dict_from_array(actions):
    return {
        "Arm1_to_Arm2_Left": actions[0],
        "Ground_to_Arm1_Left": actions[1],
        "Arm1_to_Arm2_Right": actions[2],
        "Ground_to_Arm1_Right": actions[3]
    }


def get_cluster(n_parameter_servers, n_workers):
    spec = {}
    for i in range(n_parameter_servers):
        if "ps" not in spec:
            spec["ps"] = []
        spec["ps"].append("localhost:{}".format(i + 2222))
    for i in range(n_workers):
        if "worker" not in spec:
            spec["worker"] = []
        spec["worker"].append("localhost:{}".format(i + 2222 + n_parameter_servers))
    return tf.train.ClusterSpec(spec)


def tensorboard_server_func(logdir):
    os.system('tensorboard --logdir=' + logdir + '> /dev/null 2>&1')


def chromium_func():
    os.system('chromium-browser http://localhost:6006 > /dev/null 2>&1')


def tensorboard_func(logdir):
    p1 = multiprocessing.Process(target=tensorboard_server_func, args=(logdir,))
    p1.start()
    time.sleep(5)
    p2 = multiprocessing.Process(target=chromium_func)
    p2.start()
    p2.join()
    p1.terminate()
    while p1.is_alive():
        time.sleep(0.1)


def parameter_server_func(cluster, task_index):
    server = tf.train.Server(cluster, "ps", task_index)
    server.join()


def worker_model_func(args_env, args_worker, n_updates, task_index, done_event):
    env = environment.Environment(*args_env)
    worker = JointAgentWorker(task_index, done_event, env, *args_worker)
    worker.wait_for_variables_initialization()
    worker.run_model(n_updates)


def worker_rl_func(args_env, args_worker, n_updates, train_actor, task_index, done_event):
    env = environment.Environment(*args_env)
    worker = JointAgentWorker(task_index, done_event, env, *args_worker)
    worker.wait_for_variables_initialization()
    worker.run_reinforcement_learning(n_updates, train_actor)


def worker_display_func(args_env, args_worker, task_index, done_event):
    env = environment.Environment(*args_env)
    worker = JointAgentWorker(task_index, done_event, env, *args_worker)
    worker.wait_for_variables_initialization()
    worker.run_display()


class Worker:
    def __init__(self, task_index, done_event, env, cluster, logdir, discount_factor,
                 env_step_length, sequence_length):
        self.task_index = task_index
        self.cluster = cluster
        self._n_workers = self.cluster.num_tasks("worker") - 1
        self.job_name = "worker"
        self.server = tf.train.Server(cluster, self.job_name, task_index)
        self.name = "/job:{}/task:{}".format(self.job_name, task_index)
        self.device = tf.train.replica_device_setter(worker_device=self.name, cluster=cluster)
        self.env = env
        self.discount_factor = discount_factor
        self.entropy_coef = 0.01
        self.env_step_length = env_step_length
        self.sequence_length = sequence_length
        self.done_event = done_event
        self.define_networks()
        self.logdir = logdir
        # graph = tf.get_default_graph() if task_index == 0 else None
        graph = None
        self.summary_writer = tf.summary.FileWriter(self.logdir + "/worker{}".format(task_index), graph=graph)
        self.sess = tf.Session(target=self.server.target)
        if task_index == 0 and len(self.sess.run(tf.report_uninitialized_variables())) > 0:
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

    def define_reward_scale(self):
        raise NotImplementedError("This method must be overwritten.")

    def define_networks(self):
        self.define_net_dims()
        with tf.device(self.device):
            self.define_reward_scale()
            # define model
            self.define_model()
            # define reinforcement learning
            self.define_reinforcement_learning()

    def define_model(self):
        net_dim = self.model_net_dim
        self.model_inputs = tf.placeholder(shape=[None, net_dim[0]], dtype=tf.float32)
        self.model_targets = tf.placeholder(shape=[None, net_dim[-1]], dtype=tf.float32)
        prev_layer = self.model_inputs
        with tf.variable_scope("model"):
            for i, d in enumerate(net_dim[1:]):
                activation_fn = tf.nn.relu if i < len(net_dim) - 2 else None
                prev_layer = tl.fully_connected(prev_layer, d, scope="layer{}".format(i), activation_fn=activation_fn)
        self.model_outputs = prev_layer
        self.model_losses = (self.model_outputs - self.model_targets) ** 2
        self.model_loss = tf.reduce_mean(self.model_losses, name="loss")
        self.reward_scale = (1 - self.discount_factor) / self.model_loss_converges_to
        self.reward = - self.reward_scale * self.model_loss
        # self.optimizer = tf.train.RMSPropOptimizer(0.0025, 0.99, 0.0, 1e-6)
        self.global_model_step = tf.Variable(0, dtype=tf.int32)
        self.model_optimizer = tf.train.AdamOptimizer(1e-3)
        self.model_train_op = self.model_optimizer.minimize(self.model_loss, global_step=self.global_model_step)
        self.model_loss_summary = tf.summary.scalar("model/loss", self.model_loss)
        self.model_reward_summary = tf.summary.scalar("model/reward", self.reward)
        self.model_summary = tf.summary.merge([self.model_loss_summary, self.model_reward_summary])

    def define_reinforcement_learning(self):
        net_dim = self.rl_shared_net_dim + self.actor_remaining_net_dim
        self.rl_inputs = tf.placeholder(shape=[None, net_dim[0]], dtype=tf.float32, name="actor_inputs")
        self.actions = tf.placeholder(shape=[None, net_dim[-1]], dtype=tf.float32, name="actor_picked_actions")  # picked actions
        self.actor_targets = tf.placeholder(shape=[None], dtype=tf.float32, name="actor_td_error")  # TD target value
        targets = tf.expand_dims(self.actor_targets, -1)
        batch_size = tf.shape(self.rl_inputs)[0]
        prev_layer = self.rl_inputs
        with tf.variable_scope("shared"):
            for i, d in enumerate(self.rl_shared_net_dim[1:]):
                prev_layer = tl.fully_connected(prev_layer, d, scope="layer{}".format(i))
        fork_layer = prev_layer
        with tf.variable_scope("private_to_actor"):
            for i, d in enumerate(self.actor_remaining_net_dim[:-1]):
                prev_layer = tl.fully_connected(prev_layer, d, scope="layer{}".format(i))
            self.mu = tl.fully_connected(prev_layer, self.actor_remaining_net_dim[-1], activation_fn=None, scope="mean_actions")
            self.log_sigma_2 = tl.fully_connected(prev_layer, self.actor_remaining_net_dim[-1], activation_fn=None, scope="log_variance_actions")
        self.sigma_2 = tf.exp(self.log_sigma_2)
        self.sigma = tf.exp(0.5 * self.log_sigma_2)
        self.sample_action = tf.random_normal(shape=tf.shape(self.mu)) * self.sigma + self.mu
        self.probs = 1 / (tf.sqrt(2 * pi * self.sigma_2)) * tf.exp(-(self.actions - self.mu) ** 2 / (2 * self.sigma_2))
        self.log_probs = -0.5 * (log(2 * pi) + self.log_sigma_2 + (self.actions - self.mu) ** 2 / self.sigma_2)
        self.entropy = 0.5 * (1 + log(2 * pi) + self.log_sigma_2)
        self.entropy_mean = tf.reduce_mean(self.entropy, name="entropy_mean")
        self.actor_losses = -self.log_probs * targets - self.entropy_coef * self.entropy
        self.actor_loss = tf.reduce_sum(self.actor_losses, name="loss")

        self.critic_targets = tf.placeholder(shape=[None], dtype=tf.float32, name="critic_target_returns")  # TD target value
        prev_layer = fork_layer
        with tf.variable_scope("private_to_critic"):
            for i, d in enumerate(self.critic_remaining_net_dim):
                activation_fn = tf.nn.relu if i < len(self.critic_remaining_net_dim) - 1 else None
                prev_layer = tl.fully_connected(prev_layer, d, scope="layer{}".format(i), activation_fn=activation_fn)
            self.critic_value = tf.squeeze(prev_layer, axis=1, name="critic_value")
        self.critic_losses = (self.critic_value - self.critic_targets) ** 2
        self.critic_loss = tf.reduce_mean(self.critic_losses, name="loss")
        # self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.global_rl_step = tf.Variable(0, dtype=tf.int32)
        self.critic_optimizer = tf.train.AdamOptimizer(1e-3)
        self.critic_train_op = self.critic_optimizer.minimize(self.critic_loss, global_step=self.global_rl_step)
        self.rl_optimizer = tf.train.AdamOptimizer(1e-3)
        self.rl_train_op = self.rl_optimizer.minimize(0.001 * self.actor_loss + self.critic_loss, global_step=self.global_rl_step)
        self.actor_loss_summary = tf.summary.scalar("actor/loss", self.actor_loss)
        self.critic_loss_summary = tf.summary.scalar("critic/loss", self.critic_loss)
        self.rl_reward = tf.placeholder(shape=[None], dtype=tf.float32)
        self.rl_reward_summary = tf.summary.scalar("critic/reward", tf.reduce_mean(self.rl_reward))
        self.rl_summary = tf.summary.merge([self.actor_loss_summary, self.critic_loss_summary, self.rl_reward_summary])

    def wait_for_variables_initialization(self):
        while len(self.sess.run(tf.report_uninitialized_variables())) > 0:
            print("{}  waiting for variable initialization...".format(self.name))
            time.sleep(1)

    def environment_step(self):
        for _ in range(self.env_step_length):
            self.env.step()

    def run_reinforcement_learning(self, n_updates, train_actor=True):
        global_rl_step = self.sess.run(self.global_rl_step)
        n_updates += global_rl_step
        while global_rl_step < n_updates - self._n_workers:
            # Collect some experience
            transitions = self.run_n_rl_steps()
            # Update the global networks
            global_rl_step = self.update_reinforcement_learning(*transitions, train_actor=train_actor)
        self.done_event.set()
        self.server.join()

    def run_display(self):
        win = viewer.JointAgentWindow(self.discount_factor, return_lookback=300)
        while not self.done_event.is_set():
            self.run_n_display_steps(win)
        win.close()
        self.server.join()

    def run_model(self, n_updates):
        global_model_step = self.sess.run(self.global_model_step)
        n_updates += global_model_step
        while global_model_step < n_updates - self._n_workers:
            # Collect some experience
            states = self.run_n_model_steps()
            # Update the global networks
            global_model_step = self.update_model(states)
        self.done_event.set()
        self.server.join()

    def get_action(self):
        rl_state = self.get_rl_state()
        feed_dict = self.to_rl_feed_dict(states=[rl_state])
        action = self.sess.run(self.sample_action, feed_dict=feed_dict)
        return action[0]

    def run_n_rl_steps(self):
        states = []
        actions = []
        rewards = []
        visions = []
        t_state = 0
        t_actor = 0
        t_reward = 0
        t_env = 0
        for _ in range(self.sequence_length):
            # get vision
            visions.append(self.env.vision)
            # get state
            t0 = time.time()
            model_state = self.get_model_state()
            states.append(self.get_rl_state())
            # get action
            t1 = time.time()
            action = self.get_action()
            actions.append(action)
            # set action
            t2 = time.time()
            action_dict = actions_dict_from_array(action)
            self.env.set_positions(action_dict)
            # run environment step
            self.environment_step()
            # get next state
            model_state_next = self.get_model_state()
            # get reward
            t3 = time.time()
            feed_dict = self.to_model_feed_dict(states=[model_state, model_state_next], targets=True)
            reward = self.sess.run(self.reward, feed_dict=feed_dict)
            rewards.append(reward)
            t4 = time.time()
            t_state += 1000 * (t1 - t0)
            t_actor += 1000 * (t2 - t1)
            t_env += 1000 * (t3 - t2)
            t_reward += 1000 * (t4 - t3)
        # print(self.name + " profiling run_n_model_steps: get_state {:.1f}ms  actor {:.1f}ms  reward {:.1f}ms  env {:.1f}ms".format(t_state, t_actor, t_reward, t_env))
        return states, actions, rewards

    def run_n_model_steps(self):
        states = []
        t_state = 0
        t_actor = 0
        t_env = 0
        for _ in range(self.sequence_length):
            # get state
            t0 = time.time()
            states.append(self.get_model_state())
            # get action
            t1 = time.time()
            action = self.get_action()
            # run action in env
            t2 = time.time()
            action_dict = actions_dict_from_array(action)
            self.env.set_positions(action_dict)
            self.environment_step()
            t3 = time.time()
            t_state += 1000 * (t1 - t0)
            t_actor += 1000 * (t2 - t1)
            t_env += 1000 * (t3 - t2)
        # print(self.name + " profiling run_n_model_steps: get_state {:.1f}ms  actor {:.1f}ms  env {:.1f}ms".format(t_state, t_actor, t_env))
        return states

    # TODO put that function in the subclass
    def run_n_display_steps(self, win):
        action_return_fetches = [self.sample_action, self.critic_value]
        predicted_positions_reward_fetches = [self.model_outputs, self.reward]
        for _ in range(self.sequence_length):
            # get current vision
            vision = self.env.vision
            # get current positions
            current_positions = self.env.discrete_positions
            # get action and predicted return
            rl_states = [self.get_rl_state()]
            model_states = [self.get_model_state()]
            rl_feed_dict = self.to_rl_feed_dict(states=rl_states)
            actions, predicted_returns = self.sess.run(action_return_fetches, feed_dict=rl_feed_dict)
            action = actions[0]
            predicted_return = predicted_returns[0]
            # run action in env
            action_dict = actions_dict_from_array(action)
            self.env.set_positions(action_dict)
            self.environment_step()
            # get current positions
            next_positions = self.env.discrete_positions
            # get predicted positions and reward
            model_states.append(self.get_model_state())
            model_feed_dict = self.to_model_feed_dict(states=model_states, targets=True)
            predicted_positions, current_reward = self.sess.run(predicted_positions_reward_fetches, feed_dict=model_feed_dict)
            predicted_positions = predicted_positions[0].reshape((4, -1))
            # display
            win(vision, current_positions, predicted_positions, next_positions, current_reward, predicted_return)

    def rewards_to_return(self, rewards, prev_return):
        returns = np.zeros_like(rewards)
        for i in range(len(rewards) - 1, -1, -1):
            r = rewards[i]
            prev_return = r + self.discount_factor * prev_return
            returns[i] = prev_return
        return returns

    def update_reinforcement_learning(self, states, actions, rewards, train_actor=True):
        feed_dict = self.to_rl_feed_dict(states=states)
        predicted_returns = self.sess.run(self.critic_value, feed_dict=feed_dict)
        feed_dict = self.to_rl_feed_dict(
            states=states,
            actions=actions,
            rewards=rewards,
            predicted_returns=predicted_returns)
        train_op = self.rl_train_op if train_actor else self.critic_train_op
        fetches = [self.actor_loss, self.critic_loss, self.global_rl_step, train_op, self.rl_summary]
        aloss, closs, global_rl_step, _, rl_summary = self.sess.run(fetches, feed_dict=feed_dict)
        self.summary_writer.add_summary(rl_summary, global_step=global_rl_step)
        if global_rl_step < 300 and global_rl_step % 10 == 0:
            self.summary_writer.flush()
        print("{} finished update number {} (actor loss = {:.3f}     critic loss = {:.3f})".format(
              self.name, global_rl_step, aloss, closs))
        return global_rl_step

    def update_model(self, states):
        feed_dict = self.to_model_feed_dict(states=states, targets=True)
        fetches = [self.model_loss, self.global_model_step, self.model_train_op, self.model_summary]
        loss, global_model_step, _, model_summary = self.sess.run(fetches, feed_dict=feed_dict)
        self.summary_writer.add_summary(model_summary, global_step=global_model_step)
        if global_model_step < 300 and global_model_step % 10 == 0:
            self.summary_writer.flush()
        print("{} finished update number {} (loss = {:.3f})".format(self.name, global_model_step, loss))
        return global_model_step


class JointAgentWorker(Worker):
    def get_model_state(self):
        return self.env.discrete_positions, self.env.discrete_speeds, self.env.discrete_target_positions

    def get_rl_state(self):
        return self.env.discrete_positions, self.env.discrete_speeds, self.env.discrete_target_positions

    def to_rl_feed_dict(self, states=None, actions=None, rewards=None, predicted_returns=None):
        # transforms the inputs into a feed dict for the actor
        feed_dict = {}
        if states is not None:
            np_states = np.array(states)
            new_shape = (-1, np.prod(np_states.shape[1:]))
            feed_dict[self.rl_inputs] = np.reshape(np_states, new_shape)
        if actions is not None:
            np_actions = np.array(actions)
            feed_dict[self.actions] = np_actions
        if rewards is not None and predicted_returns is not None:
            # reverse pass through the rewards here...
            returns = self.rewards_to_return(rewards, predicted_returns[-1])
            feed_dict[self.rl_reward] = rewards
            feed_dict[self.critic_targets] = returns
            feed_dict[self.actor_targets] = returns - predicted_returns
        return feed_dict

    def to_model_feed_dict(self, states, targets=False):
        feed_dict = {}
        np_states = np.array(states)
        new_shape_all = (-1, np.prod(np_states.shape[1:]))
        new_shape_target = (-1, np.prod(np_states[:, 0].shape[1:]))
        if targets:
            feed_dict[self.model_inputs] = np.reshape(np_states[:-1], new_shape_all)
            feed_dict[self.model_targets] = np.reshape(np_states[1:, 0], new_shape_target)
        else:
            feed_dict[self.model_inputs] = np.reshape(np_states, new_shape_all)
        return feed_dict

    def define_reward_scale(self):
        self.model_loss_converges_to = 0.010

    def define_net_dims(self):
        dummy_state = np.array(self.get_model_state())
        inp_dim = dummy_state.flatten().shape[0]
        model_out_dim = dummy_state[0].flatten().shape[0]
        self.model_net_dim = [inp_dim, 600, 600, 600, model_out_dim]
        self.rl_shared_net_dim = [inp_dim, 600]
        self.actor_remaining_net_dim = [4]
        self.critic_remaining_net_dim = [1]


if __name__ == "__main__":
    import environment

    # entropy_coef = 0.01

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

    discount_factor = 0.85
    min_global_updates = 350
    env_step_length = 15
    sequence_length = 1024  # 64
    logdir = TemporaryDirectory()

    N_WORKERS = 8
    N_PARAMETER_SERVERS = 2

    cluster = get_cluster(N_PARAMETER_SERVERS, N_WORKERS)
    args_env = (json_model, skin_order, skin_resolution, xlim, ylim, dpi, dt, n_discrete)
    args_worker = (cluster, logdir.name, discount_factor, env_step_length, sequence_length)


    # Start TensorBoard
    tensorboard_process = multiprocessing.Process(
        target=tensorboard_func,
        args=(logdir.name, )
    )
    tensorboard_process.start()

    # Define parameter servers processes
    parameter_servers_processes = []
    for i in range(N_PARAMETER_SERVERS):
        parameter_servers_processes.append(multiprocessing.Process(
            target=parameter_server_func,
            args=(cluster, i),
            daemon=True
        ))

    # Define workers processes
    workers_processes = []
    workers_events = []
    for i in range(N_WORKERS - 1):
        done_event = multiprocessing.Event()
        workers_events.append(done_event)
        workers_processes.append(multiprocessing.Process(
            target=worker_model_func,
            args=(args_env, args_worker, min_global_updates, i, done_event),
            daemon=True
        ))

    # Define the display process
    dpi = 3
    args_env = (json_model, skin_order, skin_resolution, xlim, ylim, dpi, dt, n_discrete)
    display_done_event = multiprocessing.Event()
    display_process = multiprocessing.Process(
        target=worker_display_func,
        args=(args_env, args_worker, N_WORKERS - 1, display_done_event),
        daemon=True
    )

    all_processes = parameter_servers_processes + workers_processes + [display_process]
    # Start all processes
    for p in all_processes:
        p.start()

    # Wait for the workers
    for done_event in workers_events:
        done_event.wait()

    # Terminate the model workers
    for p in workers_processes:
        p.terminate()
        while p.is_alive():
            print("Process still alive... waiting")
            time.sleep(0.1)

    # Define workers processes
    dpi = 1
    args_env = (json_model, skin_order, skin_resolution, xlim, ylim, dpi, dt, n_discrete)
    train_actor = False
    workers_processes = []
    workers_events = []
    for i in range(N_WORKERS - 1):
        done_event = multiprocessing.Event()
        workers_events.append(done_event)
        workers_processes.append(multiprocessing.Process(
            target=worker_rl_func,
            args=(args_env, args_worker, min_global_updates, train_actor, i, done_event),
            daemon=True
        ))

    # Start the reinforcement learning workers
    for p in workers_processes:
        p.start()

    # Wait for the workers
    for done_event in workers_events:
        done_event.wait()

    # Terminate the model workers
    for p in workers_processes:
        p.terminate()
        while p.is_alive():
            time.sleep(0.1)

    train_actor = True
    workers_processes = []
    workers_events = []
    for i in range(N_WORKERS - 1):
        done_event = multiprocessing.Event()
        workers_events.append(done_event)
        workers_processes.append(multiprocessing.Process(
            target=worker_rl_func,
            args=(args_env, args_worker, min_global_updates, train_actor, i, done_event),
            daemon=True
        ))

    # Start the reinforcement learning workers
    for p in workers_processes:
        p.start()

    # Wait for the workers
    for done_event in workers_events:
        done_event.wait()

    # Signal the display process
    display_done_event.set()

    # Terminate all processes
    # Terminate the model workers
    for p in all_processes:
        p.terminate()
        while p.is_alive():
            time.sleep(0.1)
    # tensorboard_process.join()
