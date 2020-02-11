import sys
import pickle
import environment
from replay_buffer import Buffer
import socket
import multiprocessing
import subprocess
import numpy as np
import tensorflow as tf
import time
from display import Display
import os
import agency
from goal_buffer import GoalBuffer


def pprint(tree, _indent=0):
    for key in tree:
        if key == "childs":
            pprint(tree["childs"], _indent=_indent + 4)
        else:
            print(" " * _indent + "{}".format(key))


def merge_before_after(transition_0, transition_1):
    ret = {}
    for key in transition_0:
        if key == "childs":
            merged_childs = [merge_before_after(t0, t1) for t0, t1 in zip(transition_0[key], transition_1[key])]
            ret[key] = merged_childs
        else:
            merged_0 = transition_0[key]
            merged_1 = transition_1[key]
            for subkey in list(merged_0):
                merged_0[subkey.replace("0", "1")] = merged_1[subkey]
            ret[key] = merged_0
    return ret


def stack_transitions(transitions):
    ret = {}
    transition = transitions[0]
    for key in transition:
        if key == "childs":
            stacked_childs = [stack_transitions([t[key][i] for t in transitions]) for i in range(len(transition[key]))]
            ret[key] = stacked_childs
        else:
            ret[key] = {}
            subret = ret[key]
            stacked = transition[key]
            for subkey in stacked:
                subret[subkey] = np.concatenate([t[key][subkey] for t in transitions], axis=0)
    return ret


def placeholder_tree_copy(placeholder_tree):
    ret = {}
    for key in placeholder_tree:
        if key == "childs":
            ret[key] = [placeholder_tree_copy(c) for c in placeholder_tree[key]]
        else:
            ret[key] = placeholder_tree[key].copy()
    return ret


def fill_placeholder_tree(placeholder_tree, stacked_transitions):
    for key in placeholder_tree:
        if key == "childs":
            for placeholder_tree_child, stacked_transitions_child in zip(placeholder_tree[key], stacked_transitions[key]):
                fill_placeholder_tree(placeholder_tree_child, stacked_transitions_child)
        else:
            for placeholder, val in placeholder_tree[key].items():
                placeholder_tree[key][placeholder] = stacked_transitions[key][val]


def flatten_placeholder_tree(placeholder_tree):
    ret = {}
    for key in placeholder_tree:
        if key == "childs":
            for child in placeholder_tree[key]:
                ret.update(flatten_placeholder_tree(child))
        else:
            ret.update(placeholder_tree[key])
    return ret


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


class Worker:
    def __init__(self, task_index, pipe, summary_queue, cluster, logdir, conf):
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
        agency_name = self.conf.worker_conf.agency_conf_path.split("/")[-1]
        self.discount_factor = self.conf.worker_conf.discount_factor
        self.sequence_length = self.conf.worker_conf.sequence_length
        self.replay_buffer_size = self.conf.worker_conf.buffer_size
        self.updates_per_episode = self.conf.worker_conf.updates_per_episode
        self.batch_size = self.conf.worker_conf.batch_size
        self.learning_rate = self.conf.worker_conf.learning_rate
        self.actor_speed_ratio = self.conf.worker_conf.actor_speed_ratio
        self.train_actor_every = self.conf.worker_conf.train_actor_every
        self.train_state_every = self.conf.worker_conf.train_state_every
        self.tau = self.conf.worker_conf.tau
        self.behaviour_noise_scale = self.conf.worker_conf.behaviour_noise_scale
        self.target_smoothing_noise_scale = self.conf.worker_conf.target_smoothing_noise_scale

        # everything related to the goal buffer (used for sampling goals)
        self.goals_buffer = GoalBuffer(
            self.conf.worker_conf.goal_buffer_size,
            len(self.get_gstate()[0])
        )
        self.goals_buffer.register_one(self.get_gstate()[0], self.env.ploting_data)  # initialize the buffer with 1 goal
        #
        self.pipe = pipe
        self.summary_queue = summary_queue
        self.replay_buffer = Buffer(self.replay_buffer_size)
        state_size = self.get_state().shape[1]
        goal_size = self.get_gstate().shape[1]

        self.parent_goal_0 = tf.placeholder(shape=(None, goal_size), dtype=tf.float32)
        self.parent_state_0 = tf.placeholder(shape=(None, state_size), dtype=tf.float32)
        self.parent_gstate_0 = tf.placeholder(shape=(None, goal_size), dtype=tf.float32)
        self.parent_goal_1 = tf.placeholder(shape=(None, goal_size), dtype=tf.float32)
        self.parent_state_1 = tf.placeholder(shape=(None, state_size), dtype=tf.float32)
        self.parent_gstate_1 = tf.placeholder(shape=(None, goal_size), dtype=tf.float32)

        self.agency_model = agency.AgencyRootModel.from_conf(self.conf.worker_conf.agency_conf_path)
        print("{} agency_behaviour call".format(self.name))
        self.agency_behaviour = self.agency_model(
            self.parent_goal_0,
            self.parent_state_0,
            self.parent_gstate_0,
            self.parent_goal_1,
            self.parent_state_1,
            self.parent_gstate_1,
            learning_rate=self.learning_rate,
            actor_speed_ratio=self.actor_speed_ratio,
            discount_factor=self.discount_factor,
            tau=self.tau,
            behaviour_noise_scale=self.behaviour_noise_scale,
            target_smoothing_noise_scale=None,
            batchnorm_training=False
        )
        print("{} agency_test call".format(self.name))
        self.agency_test = self.agency_model(
            self.parent_goal_0,
            self.parent_state_0,
            self.parent_gstate_0,
            self.parent_goal_1,
            self.parent_state_1,
            self.parent_gstate_1,
            learning_rate=self.learning_rate,
            actor_speed_ratio=self.actor_speed_ratio,
            discount_factor=self.discount_factor,
            tau=self.tau,
            behaviour_noise_scale=None,
            target_smoothing_noise_scale=None,
            batchnorm_training=False
        )
        print("{} agency_update call".format(self.name))
        self.agency_update = self.agency_model(
            self.parent_goal_0,
            self.parent_state_0,
            self.parent_gstate_0,
            self.parent_goal_1,
            self.parent_state_1,
            self.parent_gstate_1,
            learning_rate=self.learning_rate,
            actor_speed_ratio=self.actor_speed_ratio,
            discount_factor=self.discount_factor,
            tau=self.tau,
            behaviour_noise_scale=None,
            target_smoothing_noise_scale=self.target_smoothing_noise_scale,
            batchnorm_training=True
        )

        def get_placeholder(agency_call):
            return {
                agency_call.placeholder_goal_0: "goal_0",
                agency_call.placeholder_state_0: "state_0",
                agency_call.placeholder_gstate_0: "gstate_0",
                agency_call.placeholder_goal_1: "goal_1",
                agency_call.placeholder_state_1: "state_1",
                agency_call.placeholder_gstate_1: "gstate_1"
            }
        self._placeholder_tree_template = self.agency_update.tree_map(get_placeholder)
        self._placeholder_tree_template[self.agency_update.name] = {
            self.parent_goal_0: "root_goal_0",
            self.parent_state_0: "root_state_0",
            self.parent_gstate_0: "root_gstate_0",
            self.parent_goal_1: "root_goal_1",
            self.parent_state_1: "root_state_1",
            self.parent_gstate_1: "root_gstate_1"
        }

        # define fetches to be called in a session
        self.global_step = tf.Variable(0, dtype=tf.int64)
        self.global_step_inc = self.global_step.assign_add(1)
        self._local_step = 0
        self.train_actor_critic_fetches = {
            "root_actor_train_op": self.agency_update.root_actor_train_op,
            "root_critic_train_op": self.agency_update.root_critic_train_op,
            "root_readout_train_op": self.agency_update.root_readout_train_op,
            "root_update_target_train_op": self.agency_update.root_update_target_train_op,
            "root_batchnorm_train_op": self.agency_update.root_batchnorm_train_op,
            "global_step": self.global_step_inc
        }
        self.train_critic_fetches = {
            "root_critic_train_op": self.agency_update.root_critic_train_op,
            "root_readout_train_op": self.agency_update.root_readout_train_op,
            "root_update_target_train_op": self.agency_update.root_update_target_train_op,
            "root_batchnorm_train_op": self.agency_update.root_batchnorm_train_op,
            "global_step": self.global_step_inc
        }
        self.train_state_fetches = self.agency_update.root_state_train_op

        def behaviour_fetches_map_func(agency_call):
            return {
                "goal_0": agency_call.goal_0,
                "state_0": agency_call.state_0,
                "gstate_0": agency_call.gstate_0
            }
        self.training_behaviour_fetches = self.agency_behaviour.tree_map(behaviour_fetches_map_func)
        self.training_behaviour_fetches[self.agency_behaviour.name] = {
            "root_goal_0": self.parent_goal_0,
            "root_state_0": self.parent_state_0,
            "root_gstate_0": self.parent_gstate_0
        }
        self.testing_behaviour_fetches = self.agency_test.tree_map(behaviour_fetches_map_func)
        self.testing_behaviour_fetches[self.agency_test.name] = {
            "root_goal_0": self.parent_goal_0,
            "root_state_0": self.parent_state_0,
            "root_gstate_0": self.parent_gstate_0
        }

        def display_fetches_map_func(agency_call):
            return {
                "goal_0": agency_call.goal_0,
                "readout_goal": agency_call.readout_goal,
                "readout_state": agency_call.readout_state,
                "readout_gstate": agency_call.readout_gstate,
                "reward": agency_call.reward,
                "mean_distance_to_goal": agency_call.mean_distance_to_goal,
                "predicted_return": agency_call.predicted_return_00,
                # "predicted_return": agency_call.predicted_return_01_target,
                "critic_target": agency_call.critic_target
            }
        self.display_testing_behaviour_fetches = self.agency_test.tree_map(display_fetches_map_func)
        self.display_testing_behaviour_fetches[self.agency_test.name] = {
            "goal_0": self.parent_goal_0,
            "state_0": self.parent_state_0,
            "gstate_0": self.parent_gstate_0
        }
        self.display_training_behaviour_fetches = self.agency_behaviour.tree_map(display_fetches_map_func)
        self.display_training_behaviour_fetches[self.agency_behaviour.name] = {
            "goal_0": self.parent_goal_0,
            "state_0": self.parent_state_0,
            "gstate_0": self.parent_gstate_0
        }

        self.saver = tf.train.Saver()
        self._initializer = tf.global_variables_initializer()
        self._report_non_initialized = tf.report_uninitialized_variables()
        self.sess = tf.Session(target=self.server.target)

    def to_action(self, transition):
        if not len(transition["childs"]):
            name = [k for k in transition if k != "childs"][0]
            return {name: transition[name]["goal_0"][0, 0]}
        else:
            return {k: v for d in [self.to_action(c) for c in transition["childs"]] for k, v in d.items()}

    def add_summary(self, summary, global_step):
        try:
            self.summary_queue.put((summary, global_step), block=False)
        except multiprocessing.Queue.Full:
            print("{} could not register it's summary. (Queue is full)")

    def initialize(self):
        self.sess.run(self._initializer)
        self.sess.run(self.agency_update.root_init_target_op)
        print("{} variables initialized".format(self.name))

    def wait_for_variables_initialization(self):
        while len(self.sess.run(self._report_non_initialized)) > 0:
            print("{} waiting for variable initialization...".format(self.name))
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

    def run_training(self, n_updates):
        global_step = self.sess.run(self.global_step)
        n_updates += global_step
        while global_step < n_updates - self._n_workers:
            # Collect some experience
            transitions = self.run_n_steps()
            # Update the global networks
            for i in range(self.updates_per_episode):
               transitions = self.replay_buffer.batch(self.batch_size)
               stacked_transitions = stack_transitions(transitions)
               train_actor = self._local_step % self.train_actor_every == 0 and global_step > 10000
               train_state = train_actor
               # train_state = self._local_step % self.train_state_every == 0
               global_step = self.update_reinforcement_learning(stacked_transitions, train_actor=train_actor, train_state=train_state)
               if global_step >= n_updates - self._n_workers:
                   break
        self.pipe.send("{} going IDLE".format(self.name))

    def run_display(self, training=False):
        display = Display(self, training)
        display.show()
        self.pipe.recv()  # done
        self.pipe.send("{} (display) going IDLE".format(self.name))

    def save_video(self, path, n_frames=None, length_in_sec=None, training=False):
        display = Display(self, training)
        display.save(path, n_frames, length_in_sec)
        self.pipe.send("{} saved video under {}. display going IDLE".format(self.name, path))

    def save_contact_logs(self, name):
        path = self.logdir + "/worker{}/contacts_{}.pkl".format(self.task_index, name)
        self.env.save_contact_logs(path)
        self.pipe.send("{} saved contact logs under {}".format(self.name, path))

    def get_goal(self):
        goal, _ = self.goals_buffer.sample()
        return goal[np.newaxis]

    def get_state(self):
        return np.concatenate([self.env.sincos_positions, self.env.speeds])[np.newaxis]

    def get_gstate(self):
        return self.env.sincos_positions[np.newaxis]

    def run_n_steps(self):
        goal = self.get_goal()
        state = self.get_state()
        gstate = self.get_gstate()
        transitions = []
        for iteration in range(self.sequence_length):
            feed_dict = self.behaviour_feed_dict(goal, state, gstate)
            # feed_dict feeds root with data from 'state', 'gstate' and 'goal'
            transition_0 = self.sess.run(self.training_behaviour_fetches, feed_dict=feed_dict)
            # set action
            action = self.to_action(transition_0)
            self.env.set_speeds(action)
            # run environment step
            self.env.env_step()
            # get states
            state = self.get_state()
            gstate = self.get_gstate()
            # compute values after applying action
            feed_dict = self.behaviour_feed_dict(goal, state, gstate)
            # feed_dict feeds root with data from 'state', 'gstate' and 'goal'
            transition_1 = self.sess.run(self.training_behaviour_fetches, feed_dict=feed_dict)
            # store in replay buffer
            transition = merge_before_after(transition_0, transition_1)
            transitions.append(transition)
            self.replay_buffer.incorporate(transition)
            self.goals_buffer.register_one(gstate[0], self.env.ploting_data)
        self.randomize_env()
        return transitions

    def randomize_env(self, n=None, _answer=False):
        n = 5 if n is None else n
        for i in range(n):
            action = {
              joint_name: np.random.uniform(low=joint.lowerLimit, high=joint.upperLimit) if joint.limitEnabled else
                          np.random.uniform(low=-3.14, high=3.14) for joint_name, joint in self.env.joints.items()}
            self.env.set_positions(action)
            self.env.env_step()
        if _answer:
            self.pipe.send("{} applied {} random actions".format(self.name, n))

    def fill_goal_buffer(self):
        for i in range(1000):
            print(i, end="\r")
            action = {
              joint_name: np.random.uniform(low=joint.lowerLimit, high=joint.upperLimit) if joint.limitEnabled else
                          np.random.uniform(low=-3.14, high=3.14) for joint_name, joint in self.env.joints.items()}
            self.env.set_positions(action)
            self.env.env_step()
            self.goals_buffer.register_one(self.get_gstate()[0], self.env.ploting_data)
        print("")
        self.pipe.send("{} filled goal buffer".format(self.name))

    # def randomize_env(self, n=None, _answer=False):
    #     n = 5 if n is None else n
    #     for i in range(n):
    #         action = {
    #             "left_elbow": np.random.uniform(low=-2, high=2),
    #             "left_shoulder": np.random.uniform(low=-3.14, high=3.14),
    #             "right_elbow": np.random.uniform(low=-2, high=2),
    #             "right_shoulder": np.random.uniform(low=-3.14, high=3.14)
    #         }
    #         self.env.set_positions(action)
    #         # self.env.set_speeds(action)
    #         self.env.env_step()
    #     if _answer:
    #         self.pipe.send("{} applied {} random actions".format(self.name, n))

    def update_reinforcement_learning(self, stacked_transitions, train_actor=True, train_state=True):
        # stack_transitions as the form :
        # {"all_body": {"goal_0": data, "goal_1": data, "state_0": data, ...},
        #  "childs": [{"left_arm": {data},
        #              "childs": [...]},
        #              "right_arm": {data},
        #              "childs": [...]]
        # }
        feed_dict = self.training_feed_dict(stacked_transitions)
        # feed_dict feeds every agent with data from 'stacked_transitions'
        if train_actor:
            # train actor and critic
            ret = self.sess.run(self.train_actor_critic_fetches, feed_dict=feed_dict)
        else:
            # train critic only
            ret = self.sess.run(self.train_critic_fetches, feed_dict=feed_dict)
            # debug
            # ret, debug_data = self.sess.run([self.train_critic_fetches, self.debug_fetch], feed_dict=feed_dict)
            # print(debug_data)
        if train_state:
            # train state
            feed_dict = self.state_feed_dict(stacked_transitions)
            # feed_dict feeds only root with data from 'stacked_transitions'
            self.sess.run(self.train_state_fetches, feed_dict=feed_dict)
        if ret["global_step"] % 100 == 0:
            print("{} finished update number {}".format(self.name, ret["global_step"]))
        self._local_step += 1
        global_step = ret["global_step"]
        if (global_step < 20000 and global_step % 100 == 0) or (global_step >= 20000 and global_step % 500 == 0) or (self._local_step < 20 and global_step % 5 == 0):
            feed_dict = self.state_feed_dict(stacked_transitions)
            self.add_summary(
                self.sess.run(self.agency_update.root_summary_op, feed_dict=feed_dict),
                global_step=global_step
            )
        return global_step

    def _get_new_placeholder_tree(self):
        return placeholder_tree_copy(self._placeholder_tree_template)

    def training_feed_dict(self, stacked_transitions):
        placeholder_tree = self._get_new_placeholder_tree()
        fill_placeholder_tree(placeholder_tree, stacked_transitions)
        return flatten_placeholder_tree(placeholder_tree)

    def state_feed_dict(self, stacked_transitions):
        # hard coded, must be changed
        return {
            self.parent_goal_0: stacked_transitions["all_body"]["root_goal_0"],
            self.parent_state_0: stacked_transitions["all_body"]["root_state_0"],
            self.parent_gstate_0: stacked_transitions["all_body"]["root_gstate_0"],
            # self.parent_goal_1: stacked_transitions["all_body"]["root_goal_1"],
            self.parent_state_1: stacked_transitions["all_body"]["root_state_1"],
            self.parent_gstate_1: stacked_transitions["all_body"]["root_gstate_1"]
        }

    def display_feed_dict(self, goal_0, state_0, gstate_0, state_1, gstate_1):
        return {
            self.parent_goal_0: goal_0,
            self.parent_state_0: state_0,
            self.parent_gstate_0: gstate_0,
            self.parent_state_1: state_1,
            self.parent_gstate_1: gstate_1
        }

    def behaviour_feed_dict(self, goal, state, gstate):
        return {
            self.parent_goal_0: goal,
            self.parent_state_0: state,
            self.parent_gstate_0: gstate
        }


def collect_summaries(queue, path):
    total_time_getting = 0
    total_time_writing = 0
    last_time_printing = time.time()
    with tf.summary.FileWriter(path) as writer:
        while True:
            t0 = time.time()
            summary, global_step = queue.get()
            t1 = time.time()
            writer.add_summary(summary, global_step=global_step)
            t2 = time.time()
            total_time_getting += t1 - t0
            total_time_writing += t2 - t1
            if t2 - last_time_printing > 120:
                total = total_time_getting + total_time_writing
                print("SUMMARY COLLECTOR: {:.2f}% getting, {:.2f}% writing. Size: {}".format(
                    100 * total_time_getting / total, 100 * total_time_writing / total, queue.qsize())
                )
                last_time_printing = t2


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
        self.summary_queue = multiprocessing.Queue(maxsize=1000)
        self.summary_collector_process = multiprocessing.Process(
            target=collect_summaries,
            args=(self.summary_queue, self.logdir),
            daemon=True
        )
        ### start all processes ###
        all_processes = self.parameter_servers_processes + self.workers_processes + [self.summary_collector_process]
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
        worker = Worker(task_index, self.there_pipes[task_index], self.summary_queue, self.cluster, self.logdir, self.conf)
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

    def randomize_env(self, n=None):
        for p in self.here_worker_pipes:
            p.send(("randomize_env", n, True))
        for p in self.here_worker_pipes:
            p.recv()

    def fill_goal_buffer(self):
        for p in self.here_worker_pipes:
            p.send(("fill_goal_buffer", ))
        for p in self.here_worker_pipes:
            p.recv()

    def asynchronously_train(self, n_updates):
        for p in self.here_worker_pipes:
            p.send(("run_training", n_updates))
        for p in self.here_worker_pipes:
            p.recv()

    def save_model(self):
        self.here_worker_pipes[0].send(("save", self.checkpointsdir))
        print(self.here_worker_pipes[0].recv())

    def save_video(self, name, path=None, n_frames=None, length_in_sec=None, training=False):
        path = self.videodir + "/" + name if path is None else path + "/" + name
        self.here_worker_pipes[0].send(("save_video", path, n_frames, length_in_sec, training))
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
