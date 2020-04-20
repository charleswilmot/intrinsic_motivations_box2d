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
        self.sequence_length = self.conf.worker_conf.sequence_length  # to delete (from here and elswhere)
        self.replay_buffer_size = self.conf.worker_conf.buffer_size
        self.updates_per_episode = self.conf.worker_conf.updates_per_episode
        self.time_scale_factor = self.conf.worker_conf.time_scale_factor
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

        names = self.agency_update.map_level(lambda agency_call: agency_call.name)
        self.last_level = len(names)
        self.last_level_names = names[-1]

        def get_behaviour_fetch(agency_call):
            return {
                "parent_goal_0": agency_call.parent_goal_0,
                "parent_state_0": agency_call.parent_state_0,
                "parent_gstate_0": agency_call.parent_gstate_0,
                "goal_0": agency_call.goal_0,
                "state_0": agency_call.state_0,
                "gstate_0": agency_call.gstate_0
            }

        ### Training behaviour fetches
        per_level_per_agent_behaviour_fetches = self.agency_behaviour.map_level(get_behaviour_fetch)
        self.training_behaviour_fetches = []
        for level_behaviour_fetches in per_level_per_agent_behaviour_fetches:
            new_level_behaviour_fetches = {
                key: tf.concat(
                    [agent_behaviour_fetches[key] for agent_behaviour_fetches in level_behaviour_fetches],
                    axis=0)
                    for key in level_behaviour_fetches[0]
            }
            self.training_behaviour_fetches.append(new_level_behaviour_fetches)
        ### self.training_behaviour_fetches = [
        #    {
        #        "parent_goal_0": tensor of shape (n_agent_at_that_level, parent_goal_0_size),
        #        "parent_state_0": tensor of shape (n_agent_at_that_level, parent_state_0_size),
        #        ...
        #    }
        # for every level]

        ### Testing behaviour fetches
        per_level_per_agent_behaviour_fetches = self.agency_test.map_level(get_behaviour_fetch)
        self.testing_behaviour_fetches = []
        for level_behaviour_fetches in per_level_per_agent_behaviour_fetches:
            new_level_behaviour_fetches = {
                key: tf.concat(
                    [agent_behaviour_fetches[key] for agent_behaviour_fetches in level_behaviour_fetches],
                    axis=0)
                    for key in level_behaviour_fetches[0]
            }
            self.testing_behaviour_fetches.append(new_level_behaviour_fetches)

        def get_placeholders(agency_call):
            return {
                "parent_goal_0": agency_call.parent_goal_0,
                "parent_state_0": agency_call.parent_state_0,
                "parent_gstate_0": agency_call.parent_gstate_0,
                "parent_state_1": agency_call.parent_state_1,  # for the summaries and training the critic
                "parent_gstate_1": agency_call.parent_gstate_1,
                "goal_0": agency_call.placeholder_goal_0,
                # "parent_goal_1": agency_call.parent_goal_1,    # a priori useless
                # "state_0": agency_call.placeholder_state_0,    # a priori useless
                # "gstate_0": agency_call.placeholder_gstate_0   # a priori useless
            }

        ### for every level: a list for every agent of dict containing all input / output placeholders
        ### the keys must match those of the stacked_transitions / behaviour_fetch (except the time indice)
        self.training_placeholder = self.agency_update.map_level(get_placeholders)

        def get_actor_critic_fetches(agency_call):
            return {
                # "readout_train_op": agency_call.readout_train_op,  # not used at the moment
                "state_train_op": agency_call.state_train_op,
                "actor_train_op": agency_call.actor_train_op,
                "critic_train_op": agency_call.critic_train_op,
                "update_target_weights_op": agency_call.update_target_weights_op,
                "batchnorm_train_op": agency_call.batchnorm_train_op,  # depends on placeholder root_state_0 ???
                "level_counter": agency_call.level_counter_inc
            }

        def get_critic_fetches(agency_call):
            return {
                # "readout_train_op": agency_call.readout_train_op,  # not used at the moment
                "critic_train_op": agency_call.critic_train_op,
                "update_target_weights_op": agency_call.update_target_weights_op,
                "batchnorm_train_op": agency_call.batchnorm_train_op,  # depends on placeholder root_state_0 ???
                "level_counter": agency_call.level_counter_inc
            }

        self.train_actor_critic_fetches = self.agency_update.map_level(get_actor_critic_fetches)
        self.train_critic_fetches = self.agency_update.map_level(get_critic_fetches)

        def get_goal_placeholders(agency_call):
            return agency_call.placeholder_goal_0

        ### list of lists of goal placeholder (per level)
        ### order in one level must match order of beheviour fetches
        self.training_behaviour_goals_placeholders = [[self.parent_goal_0]] + self.agency_behaviour.map_level(get_goal_placeholders)[:-1]
        self.testing_behaviour_goals_placeholders = [[self.parent_goal_0]] + self.agency_test.map_level(get_goal_placeholders)[:-1]
        self.summaries = list(map(tf.summary.merge, self.agency_update.map_level(lambda agency_call: agency_call.summary)))
        ##################################
        self.replay_buffers = [Buffer(self.replay_buffer_size) for level in range(self.last_level + 1)]
        self.global_step = tf.Variable(0, dtype=tf.int64)
        self.global_step_inc = self.global_step.assign_add(1)
        self._local_step = 0

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
        for i in range(self.conf.worker_conf.goal_buffer_size * 10):
            action = {
              joint_name: np.random.uniform(low=joint.lowerLimit, high=joint.upperLimit) if joint.limitEnabled else
                          np.random.uniform(low=-3.14, high=3.14) for joint_name, joint in self.env.joints.items()}
            self.env.set_positions(action)
            self.env.env_step()
            self.goals_buffer.register_one(self.get_gstate()[0], self.env.ploting_data)
        self.pipe.send("{} filled goal buffer".format(self.name))

    def register_current_gstate(self):
        gstate = self.get_gstate()
        self.goals_buffer.register_one(gstate[0], self.env.ploting_data)

    def train(self, n_updates):
        global_step = self.sess.run(self.global_step)
        global_step_limit = n_updates + global_step
        goals = [self.get_goal()]
        while not self.recursive_train_at(0, goals, global_step_limit):
            goals = [self.get_goal()]
        self.pipe.send("{} going IDLE".format(self.name))

    def apply(self, goals):
        self.env.set_speeds(dict(zip(self.last_level_names, goals)))
        self.env.env_step()
        # self.env.step()
        # print("measured speed: {}".format(self.env.speeds))

    def recursive_train_at(self, level, goals, global_step_limit):
        if level == self.last_level:
            self.apply(goals)
            self.register_current_gstate()
            return False
        else:
            must_stop = self.recursive_gather_data_at(level, goals, global_step_limit)
            if must_stop:
                return True
            for i in range(self.updates_per_episode):
                must_stop = self.train_from_buffer_at(level, global_step_limit)
                if must_stop:
                    return True
        return must_stop

    def recursive_gather_data_at(self, level, goals, global_step_limit):
        state = self.get_state()
        gstate = self.get_gstate()
        for i in range(self.time_scale_factor):
            ### feed_dict must feed root state and roo gstate, but goal in the level's goal placeholders
            if i == 0:
                feed_dict = self.behaviour_feed_dict(level, goals, state, gstate)
                transition_0 = self.sess.run(self.training_behaviour_fetches[level], feed_dict=feed_dict)  # (my level only)
            else:
                transition_0 = transition_1
            # set action
            must_stop = self.recursive_train_at(level + 1, transition_0["goal_0"], global_step_limit)
            if must_stop:
                return True
            # compute values after applying action
            ### update feed_dict
            feed_dict[self.parent_state_0] = self.get_state()
            feed_dict[self.parent_gstate_0] = self.get_gstate()
            # feed_dict feeds root with data from 'state', 'gstate' and 'goal'
            transition_1 = self.sess.run(self.training_behaviour_fetches[level], feed_dict=feed_dict)
            # store in replay buffer
            self.replay_buffers[level].incorporate(transition_0, transition_1)
        return must_stop

    def train_from_buffer_at(self, level, global_step_limit):
        transitions = self.replay_buffers[level].batch(self.batch_size)
        train_actor = self._local_step % self.train_actor_every == 0
        global_step = self.update_at(level, transitions, train_actor=train_actor)
        must_stop = global_step >= global_step_limit - self._n_workers
        return must_stop

    def update_at(self, level, transitions, train_actor=True):
        # transitions is a slice of the replay buffer :
        # array of shape (batch_size, n_agents) of dtype behaviour_fetch_type
        feed_dict = self.training_feed_dict(level, transitions)
        global_step_fetch = self.global_step_inc if level == self.last_level - 1 else self.global_step
        # feed_dict feeds every agent at one level with data from 'transitions'
        if train_actor:
            ret, global_step = self.sess.run([self.train_actor_critic_fetches[level], global_step_fetch], feed_dict=feed_dict)
        else:
            ret, global_step = self.sess.run([self.train_critic_fetches[level], global_step_fetch], feed_dict=feed_dict)
        level_counter = ret[0]["level_counter"]
        if level_counter % 100 == 0:
            print("{} finished update number {} at level {} (global_step {})".format(self.name, level_counter, level, global_step))
        self._local_step += 1
        log_freq = 5 * 10 ** int(np.log10(level_counter) / 2)
        if level_counter % log_freq == 0 or (self._local_step < 20 and level_counter % 5 == 0):
            # TODO
            # feed_dict = self.state_feed_dict(transitions)
            self.add_summary(
                self.sess.run(self.summaries[level], feed_dict=feed_dict),
                global_step=level_counter
            )
        return global_step

    def behaviour_feed_dict(self, level, goals, state, gstate, train=True):
        behaviour_goals_placeholders = self.training_behaviour_goals_placeholders[level] \
                        if train else self.testing_behaviour_goals_placeholders[level]
        feed_dict = {}
        for placeholder, goal in zip(behaviour_goals_placeholders, goals):
            feed_dict[placeholder] = goal.reshape((1, -1))
        feed_dict[self.parent_state_0] = state
        feed_dict[self.parent_gstate_0] = gstate
        return feed_dict

    def training_feed_dict(self, level, transitions):
        feed_dict = {}
        for i, placeholders in enumerate(self.training_placeholder[level]):
            for key, placeholder in placeholders.items():
                if placeholder not in feed_dict:
                    feed_dict[placeholder] = transitions[key][:, i]
        return feed_dict



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
