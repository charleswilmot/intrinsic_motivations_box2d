import threading
import queue
# from multiprocessing import Process
# from subprocess import Popen
import numpy as np
from numpy import pi, log
from numpy.random import normal
import tensorflow as tf
import tensorflow.contrib.layers as tl
import time
import viewer


class profile:
    def __init__(self, string):
        self.string = string

    def __enter__(self):
        self.start = time.time()
        # print(self.string)

    def __exit__(self, type, value, traceback):
        # print(self.string, "  done in {:.2f}ms".format(1000 * (time.time() - self.start)))
        pass


class GraphTree:
    def __init__(self, name, *conf, **kwconf):
        self._conf = conf
        self._kwconf = kwconf
        self.name = name
        self.update_count = tf.Variable(0, dtype=tf.int32, name="global_update_count_{}".format(self.name))

    def get_download_vars_op(self, other):
        update_ops = [v2.assign(v1) for v1, v2 in zip(other.variables, self.variables)]
        return tf.group(*update_ops)

    def get_train_and_upload_vars_op(self, other):
        self.inc_update_count = other.update_count.assign_add(1)
        self.reset_update_count = other.update_count.assign(0)
        local_grads, _ = zip(*self.grads_vars)
        local_grads, _ = tf.clip_by_global_norm(local_grads, 5.0)
        _, global_vars = zip(*other.grads_vars)
        local_grads_global_vars = list(zip(local_grads, global_vars))
        update = other.optimizer.apply_gradients(local_grads_global_vars)
        return update

    def copy(self):
        return self.__class__(self.name, *self._conf, **self._kwconf)


class GraphNode(GraphTree):
    def __init__(self, name, *conf, **kwconf):
        super().__init__(name, *conf, **kwconf)
        with tf.variable_scope(name):
            scope = tf.get_variable_scope().name
            self._define_subgraphs(*conf, **kwconf)
            self.variables = sorted(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope), key=lambda v: v.name)
            self.grads_vars = self.optimizer.compute_gradients(self.loss)
            self.grads_vars = sorted([(g, v) for g, v in self.grads_vars if g is not None], key=lambda t: t[1].name)
        self._add_subgraphs_as_attr()

    def _define_subgraphs(self, *conf, **kwconf):
        raise NotImplementedError("Calling from abstract class")

    def _add_subgraphs_as_attr(self):
        for key in self.subgraphs:
            self.__setattr__(key, self.subgraphs[key])


class GraphLeaf(GraphTree):
    def __init__(self, name, *conf, **kwconf):
        super().__init__(name, *conf, **kwconf)
        self.collection_name = tf.get_variable_scope().name + "/{}".format(self.name)
        self.tlkwargs = {"reuse": tf.AUTO_REUSE, "variables_collections": [self.collection_name]}
        self._define_graph(*conf, **kwconf)
        self.grads_vars = self.optimizer.compute_gradients(self.loss)
        self.variables = sorted(tf.get_collection(self.collection_name), key=lambda v: v.name)
        self.grads_vars = sorted([(g, v) for g, v in self.grads_vars if g is not None], key=lambda t: t[1].name)

    def _define_graph(self, *conf, **kwconf):
        # must define self.loss and self.optimizer
        # can define as many placeholder as needed (eg self.inputs, self.state, etc)
        # all variables defined here must be stored in the collection called self.collection_name
        raise NotImplementedError("Calling from abstract class")


class A3CDiscreteActorMLP(GraphLeaf):
    def _define_graph(self, net_dim, entropy_coef):
        self.inputs = tf.placeholder(shape=[None, net_dim[0]], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)  # id of the picked action
        self.targets = tf.placeholder(shape=[None], dtype=tf.float32)  # TD target value
        batch_size = tf.shape(self.inputs)[0]
        prev_layer = self.inputs
        with tf.variable_scope("shared"):
            for i, d in enumerate(net_dim[1:-1]):
                prev_layer = tl.fully_connected(prev_layer, d, scope="layer{}".format(i), **self.tlkwargs)
        with tf.variable_scope("private_to_actor"):
            self.logits = tl.fully_connected(prev_layer, net_dim[-1], activation_fn=None, scope="output", **self.tlkwargs)
            self.probs = tf.nn.softmax(self.logits, name="probs") + 1e-8
        self.entropy = -tf.reduce_sum(self.probs * tf.log(self.probs), 1, name="entropy")
        self.entropy_mean = tf.reduce_mean(self.entropy, name="entropy_mean")
        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * net_dim[-1] + self.actions
        self.picked_action_probs = tf.gather(tf.reshape(self.probs, [-1]), gather_indices)
        # Warning + or - entropy ??? Check sign here
        self.losses = - (tf.log(self.picked_action_probs) * self.targets + entropy_coef * self.entropy)
        self.loss = tf.reduce_sum(self.losses, name="loss")
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)


class A3CContinuousActorMLP(GraphLeaf):
    def _define_graph(self, net_dim, entropy_coef):
        self.inputs = tf.placeholder(shape=[None, net_dim[0]], dtype=tf.float32, name="actor_inputs")
        self.actions = tf.placeholder(shape=[None, net_dim[-1]], dtype=tf.float32, name="actor_picked_actions")  # picked actions
        self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="actor_td_error")  # TD target value
        targets = tf.expand_dims(self.targets, -1)
        batch_size = tf.shape(self.inputs)[0]
        prev_layer = self.inputs
        with tf.variable_scope("shared"):
            for i, d in enumerate(net_dim[1:-1]):
                prev_layer = tl.fully_connected(prev_layer, d, scope="layer{}".format(i), **self.tlkwargs)
        with tf.variable_scope("private_to_actor"):
            self.mu = tl.fully_connected(prev_layer, net_dim[-1], activation_fn=None, scope="mean_actions", **self.tlkwargs)
            self.log_sigma_2 = tl.fully_connected(prev_layer, net_dim[-1], activation_fn=None, scope="log_variance_actions", **self.tlkwargs)
        self.sigma_2 = tf.exp(self.log_sigma_2)
        self.sigma = tf.exp(0.5 * self.log_sigma_2)
        self.sample = tf.random_normal(shape=tf.shape(self.mu)) * self.sigma + self.mu
        self.probs = 1 / (tf.sqrt(2 * pi * self.sigma_2)) * tf.exp(-(self.actions - self.mu) ** 2 / (2 * self.sigma_2))
        self.log_probs = -0.5 * (log(2 * pi) + self.log_sigma_2 + (self.actions - self.mu) ** 2 / self.sigma_2)
        self.entropy = 0.5 * (1 + log(2 * pi) + self.log_sigma_2)
        self.entropy_mean = tf.reduce_mean(self.entropy, name="entropy_mean")
        self.losses = -self.log_probs * targets - entropy_coef * self.entropy
        self.loss = tf.reduce_sum(self.losses, name="loss")
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)


class A3CCriticMLP(GraphLeaf):
    def _define_graph(self, net_dim):
        self.inputs = tf.placeholder(shape=[None, net_dim[0]], dtype=tf.float32, name="critic_inputs")
        self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="critic_target_returns")  # TD target value
        prev_layer = self.inputs
        with tf.variable_scope("shared"):
            for i, d in enumerate(net_dim[1:-1]):
                prev_layer = tl.fully_connected(prev_layer, d, scope="layer{}".format(i), **self.tlkwargs)
        with tf.variable_scope("private_to_critic"):
            self.logits = tl.fully_connected(prev_layer, net_dim[-1], activation_fn=None, scope="output", **self.tlkwargs)
            self.logits = tf.squeeze(self.logits, axis=1, name="logits")
        self.losses = tf.squared_difference(self.logits, self.targets)
        self.loss = tf.reduce_sum(self.losses, name="loss")
        # self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.optimizer = tf.train.AdamOptimizer()


class A3CMLP(GraphNode):
    def _define_subgraphs(self, net_dim, entropy_coef):
        actor_net_dim = net_dim
        critic_net_dim = net_dim[:-1] + [1]
        self.subgraphs = {}
        self.subgraphs["actor"] = A3CContinuousActorMLP("actor", actor_net_dim, entropy_coef)
        self.subgraphs["critic"] = A3CCriticMLP("critic", critic_net_dim)
        self.loss = self.subgraphs["actor"].loss + self.subgraphs["critic"].loss
        # self.optimizer = tf.train.RMSPropOptimizer(0, 0.1, 0.0, 1e-6)
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)


## Assumes ten has rank 2
def _discretize(ten, mini, maxi, n):
    # m = tf.shape(ten)[1]
    m = ten.get_shape().as_list()[1]
    x = tf.linspace(0.0, 1.0, n)
    x = tf.reshape(x, (1, 1, -1))
    arg = (tf.expand_dims(ten, -1) - mini) / (maxi - mini)
    x = tf.tile(x, tf.shape(arg))
    x = x - arg
    x = tf.reshape(x, (-1, n * m))
    return tf.pow(tf.cos(x * pi), 20)


class JointForwardModel(GraphLeaf):
    def _define_graph(self, net_dim):
        self.inputs = tf.placeholder(shape=[None, net_dim[0]], dtype=tf.float32)
        self.targets = tf.placeholder(shape=[None, net_dim[-1]], dtype=tf.float32)
        prev_layer = self.inputs
        with tf.variable_scope("private"):
            for i, d in enumerate(net_dim[1:]):
                prev_layer = tl.fully_connected(prev_layer, d, scope="layer{}".format(i), **self.tlkwargs)
        self.outputs = prev_layer
        # self.losses = tf.squared_difference(self.outputs, self.targets)
        self.losses = (self.outputs - self.targets) ** 2
        self.loss = tf.reduce_mean(self.losses, name="loss")
        self.optimizer = tf.train.RMSPropOptimizer(0.0025, 0.99, 0.0, 1e-6)


class JointAgent(GraphNode):
    def _define_subgraphs(self, model_net_dim, rl_net_dim, entropy_coef):
        self.subgraphs = {}
        self.subgraphs["reinforcement_learning"] = A3CMLP("reinforcement_learning", rl_net_dim, entropy_coef)
        self.subgraphs["model"] = JointForwardModel("joint_model", model_net_dim)
        self.loss = self.subgraphs["reinforcement_learning"].loss + self.subgraphs["model"].loss
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)


def actions_dict_from_array(actions):
    return {
        "Arm1_to_Arm2_Left": actions[0],
        "Ground_to_Arm1_Left": actions[1],
        "Arm1_to_Arm2_Right": actions[2],
        "Ground_to_Arm1_Right": actions[3]
    }


class Worker:
    def __init__(self, name, env, global_graph, discount_factor, min_global_updates, env_step_length, sequence_length):
        self.name = name
        self.env = env
        self.discount_factor = discount_factor
        self.min_global_updates = min_global_updates
        self.env_step_length = env_step_length
        self.sequence_length = sequence_length
        self.display_queue = queue.Queue()
        with tf.variable_scope(name):
            self.local_graph = global_graph.copy()
        local_rl_graph = self.local_graph.reinforcement_learning
        local_model_graph = self.local_graph.model
        self.download_rl_vars = local_rl_graph.get_download_vars_op(global_graph.reinforcement_learning)
        self.upload_train_rl_vars = local_rl_graph.get_train_and_upload_vars_op(global_graph.reinforcement_learning)
        self.download_model_vars = local_model_graph.get_download_vars_op(global_graph.model)
        self.upload_train_model_vars = local_model_graph.get_train_and_upload_vars_op(global_graph.model)

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

    def to_window(self, states, actions, rewards, visions, returns, predicted_rturn):
        # raise NotImplementedError("This method must be overwritten.")
        return None

    def environment_step(self):
        for _ in range(self.env_step_length):
            self.env.step()

    def run_reinforcement_learning(self, sess, coord):
        with sess.as_default(), sess.graph.as_default():
            try:
                sess.run(self.local_graph.reinforcement_learning.reset_update_count)
                while not coord.should_stop():
                    # Copy Parameters from the global networks
                    sess.run(self.download_rl_vars)
                    # Collect some experience
                    transitions = self.run_n_rl_steps(sess)
                    # Update the global networks
                    self.update_reinforcement_learning(sess, coord, *transitions)
            except tf.errors.CancelledError:
                return

    def run_model(self, sess, coord):
        with sess.as_default(), sess.graph.as_default():
            try:
                sess.run(self.local_graph.model.reset_update_count)
                while not coord.should_stop():
                    # Copy Parameters from the global networks
                    with profile(self.name + " downloading vars"):
                        sess.run(self.download_model_vars)
                    # Collect some experience
                    with profile(self.name + " run n steps ({})".format(self.sequence_length)):
                        states = self.run_n_model_steps(sess)
                    # Update the global networks
                    with profile(self.name + " uploading vars"):
                        self.update_model(sess, coord, states)
            except tf.errors.CancelledError:
                return

    def get_action(self, sess):
        rl_state = self.get_rl_state()
        feed_dict = self.to_rl_feed_dict(states=[rl_state])
        action = sess.run(self.local_graph.reinforcement_learning.actor.sample, feed_dict=feed_dict)
        return action[0]

    def run_n_rl_steps(self, sess):
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
            action = self.get_action(sess)
            actions.append(action)
            # set action
            t2 = time.time()
            action_dict = actions_dict_from_array(action)
            env.set_positions(action_dict)
            # run environment step
            self.environment_step()
            # get next state
            model_state_next = self.get_model_state()
            # get reward
            t3 = time.time()
            feed_dict = self.to_model_feed_dict(states=[model_state, model_state_next], targets=True)
            neg_reward = sess.run(self.local_graph.model.loss, feed_dict=feed_dict)
            rewards.append(- neg_reward)
            t4 = time.time()
            t_state += 1000 * (t1 - t0)
            t_actor += 1000 * (t2 - t1)
            t_env += 1000 * (t3 - t2)
            t_reward += 1000 * (t4 - t3)
        # print(self.name + " profiling run_n_model_steps: get_state {:.1f}ms  actor {:.1f}ms  reward {:.1f}ms  env {:.1f}ms".format(t_state, t_actor, t_reward, t_env))
        return states, actions, rewards, visions

    def run_n_model_steps(self, sess):
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
            action = self.get_action(sess)
            # run action in env
            t2 = time.time()
            action_dict = actions_dict_from_array(action)
            env.set_positions(action_dict)
            self.environment_step()
            t3 = time.time()
            t_state += 1000 * (t1 - t0)
            t_actor += 1000 * (t2 - t1)
            t_env += 1000 * (t3 - t2)
        # print(self.name + " profiling run_n_model_steps: get_state {:.1f}ms  actor {:.1f}ms  env {:.1f}ms".format(t_state, t_actor, t_env))
        return states

    def rewards_to_return(self, rewards, prev_return):
        returns = np.zeros_like(rewards)
        for i in range(len(rewards) - 1, -1, -1):
            r = rewards[i]
            prev_return = r + self.discount_factor * prev_return
            returns[i] = prev_return
        return returns

    def update_reinforcement_learning(self, sess, coord, states, actions, rewards, visions):
        rl_graph = self.local_graph.reinforcement_learning
        feed_dict = self.to_rl_feed_dict(states=states)
        predicted_returns = sess.run(rl_graph.critic.logits, feed_dict=feed_dict)
        returns = self.rewards_to_return(rewards, predicted_returns[-1])
        feed_dict = self.to_rl_feed_dict(states=states, actions=actions, returns=returns, predicted_returns=predicted_returns)
        fetches = [rl_graph.actor.loss, rl_graph.critic.loss, rl_graph.inc_update_count, self.upload_train_rl_vars, rl_graph.actor.log_probs]
        aloss, closs, n_model_global_updates, _, log_prob = sess.run(fetches, feed_dict=feed_dict)
        # Display environment
        self.display_queue.put(self.to_window(states, actions, rewards, visions, returns, predicted_returns))
        print("{} finished update number {} (actor loss = {:.3f}     critic loss = {:.3f})".format(self.name, n_model_global_updates, aloss, closs))
        if n_model_global_updates >= self.min_global_updates:
            coord.request_stop()

    def update_model(self, sess, coord, states):
        feed_dict = self.to_model_feed_dict(states=states, targets=True)
        fetches = [self.local_graph.model.loss, self.local_graph.model.inc_update_count, self.upload_train_model_vars]
        loss, n_model_global_updates, _ = sess.run(fetches, feed_dict=feed_dict)
        print("{} finished update number {} (loss = {:.3f})".format(self.name, n_model_global_updates, loss))
        if n_model_global_updates >= self.min_global_updates:
            coord.request_stop()


class JointAgentWorker(Worker):
    def get_model_state(self):
        return self.env.discrete_positions, self.env.discrete_speeds, self.env.discrete_target_positions

    def get_rl_state(self):
        return self.env.discrete_positions, self.env.discrete_speeds, self.env.discrete_target_positions

    def to_rl_feed_dict(self, states=None, actions=None, returns=None, predicted_returns=None):
        # transforms the inputs into a feed dict for the actor
        actor = self.local_graph.reinforcement_learning.actor
        critic = self.local_graph.reinforcement_learning.critic
        feed_dict = {}
        if states is not None:
            np_states = np.array(states)
            new_shape = (-1, np.prod(np_states.shape[1:]))
            feed_dict[actor.inputs] = np.reshape(np_states, new_shape)
            feed_dict[critic.inputs] = np.reshape(np_states, new_shape)
        if actions is not None:
            np_actions = np.array(actions)
            feed_dict[actor.actions] = np_actions
        if returns is not None and predicted_returns is not None:
            # reverse pass through the rewards here...
            feed_dict[critic.targets] = returns
            feed_dict[actor.targets] = returns - predicted_returns
        return feed_dict

    def to_model_feed_dict(self, states, targets=False):
        model = self.local_graph.model
        feed_dict = {}
        np_states = np.array(states)
        new_shape_all = (-1, np.prod(np_states.shape[1:]))
        new_shape_target = (-1, np.prod(np_states[:, 0].shape[1:]))
        if targets:
            feed_dict[model.inputs] = np.reshape(np_states[:-1], new_shape_all)
            feed_dict[model.targets] = np.reshape(np_states[1:, 0], new_shape_target)
        else:
            feed_dict[model.inputs] = np.reshape(np_states, new_shape_all)
        return feed_dict

    def to_window(self, states, actions, rewards, visions, returns, predicted_rturn):
        positions = [x[0] for x in states]
        targets = [x[2] for x in states]
        prevs = positions
        return visions, positions, targets, prevs, returns, predicted_rturn


if __name__ == "__main__":
    import environment

    model_net_dim = [12 * 32, 200, 200, 4 * 32]
    rl_net_dim = [12 * 32, 200, 200, 4]
    entropy_coef = 0.01
    with tf.variable_scope("global"):
        ja = JointAgent("joint_agent", model_net_dim, rl_net_dim, entropy_coef)



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

    with tf.device('/cpu:0'):
        workers = []
        windows = []
        for i in range(4):
            env = environment.Environment("../models/two_arms.json", skin_order, skin_resolution, xlim, ylim, dpi=1, dt=1 / 150.0)
            workers.append(JointAgentWorker("worker_{}".format(i), env, ja, 0.2, 300, 500, 64))
            win = viewer.JointAgentWindow()
            win.set_return_lim((-0.2, 0))
            windows.append(win)
        queues = [w.display_queue for w in workers]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()

        ### TRAIN MODEL
        worker_threads = []
        for worker in workers:
            def target_run_model():
                return worker.run_model(sess, coord)
            t = threading.Thread(target=target_run_model)
            t.start()
            worker_threads.append(t)
        coord.join(worker_threads)
        coord.clear_stop()

        ### TRAIN RL
        worker_threads = []
        for worker in workers:
            def target_run_rl():
                return worker.run_reinforcement_learning(sess, coord)
            t = threading.Thread(target=target_run_rl)
            t.start()
            worker_threads.append(t)
        all_queue_empty = False
        while not (coord.should_stop() and all_queue_empty):
            all_queue_empty = True
            for i in range(len(workers)):
                q = queues[i]
                if not q.empty():
                    all_queue_empty = False
                    while not q.empty():
                        data = q.get(block=False)
                    windows[i](*data)
        coord.join(worker_threads)
