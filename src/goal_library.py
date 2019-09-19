# import os
# import png
import numpy as np
import multiprocessing as mp
from discretization import np_discretization_reward
from scipy.special import softmax
import pickle
import matplotlib.pyplot as plt


class SharedArray:
    def __init__(self, shape, dtype):
        self._shape = shape
        self._dtype = dtype
        self._ctype = np.ctypeslib.as_ctypes_type(self.dtype)
        self._array = mp.Array(self.ctype, int(np.prod(self.shape)))
        self._nparray = np.frombuffer(self.array.get_obj(), dtype=self.dtype).reshape(self.shape)
        self.get_lock = self.array.get_lock

    def __getitem__(self, index):
        return self.nparray.__getitem__(index)

    def __setitem__(self, index, val):
        return self.nparray.__setitem__(index, val)

    def _get_shape(self):
        return self._shape

    def _get_dtype(self):
        return self._dtype

    def _get_ctype(self):
        return self._ctype

    def _get_array(self):
        return self._array

    def _get_nparray(self):
        return self._nparray

    shape = property(_get_shape)
    dtype = property(_get_dtype)
    ctype = property(_get_ctype)
    array = property(_get_array)
    nparray = property(_get_nparray)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_ctype"]
        del state["_array"]
        del state["get_lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class GoalLibrary:
    def __init__(self, library_size, goal_size, vision_shape, ema_speed):
        # store parameters
        self._library_size = library_size
        self._goal_size = goal_size
        self._vision_shape = vision_shape
        self._ema_speed = ema_speed
        self.snapshots = {}
        self.goal_dtype = np.dtype(
            [("intensities", np.float32, self.goal_size),
             ("r|p", np.float32),
             ("r|~p", np.float32),
             ("delta_r|p", np.float32)])
        self.vision_array = SharedArray(shape=(self.library_size,) + tuple(self.vision_shape), dtype=np.uint8)
        self.goal_array = SharedArray(shape=(self.library_size,), dtype=self.goal_dtype)
        with self.goal_array.get_lock():
            self.goal_array["r|p"] = 1
            self.goal_array["r|~p"] = 1
        self._n_goals = SharedArray(shape=(), dtype=np.int32)
        with self._n_goals.get_lock():
            self._n_goals.nparray[()] = 1

    def _get_library_size(self):
        return self._library_size

    def _get_goal_size(self):
        return self._goal_size

    def _get_vision_shape(self):
        return self._vision_shape

    def _get_ema_speed(self):
        return self._ema_speed

    def _get_n_goals(self):
        return int(self._n_goals.nparray)

    def _get_registered_goals(self):
        return self.goal_array[:self.n_goals]

    library_size = property(_get_library_size)
    goal_size = property(_get_goal_size)
    vision_shape = property(_get_vision_shape)
    ema_speed = property(_get_ema_speed)
    n_goals = property(_get_n_goals)
    registered_goals = property(_get_registered_goals)

    def register_goals(self, observed_goals, pursued_goal, visions=None):
        if visions is None:
            for goal in observed_goals:
                self.register_goal(goal, pursued_goal)
        else:
            for goal, vision in zip(observed_goals):
                self.register_goal(goal, pursued_goal, vision=vision)

    def register_goal(self, observed_goal, pursued_goal, vision=None):
        observed_goal = np.array(observed_goal, dtype=np.float32)
        pursued_goal = np.array(pursued_goal, dtype=np.float32)
        is_pursued_goal_reached = np_discretization_reward(observed_goal, pursued_goal)
        index_pursued_goal = np.where((self.goal_array["intensities"] == pursued_goal).all(axis=1))[0][0]
        before = self.goal_array[index_pursued_goal]["r|p"]
        with self.goal_array.get_lock(), self._n_goals.get_lock():
            self.goal_array[index_pursued_goal]["r|p"] *= self.ema_speed
            if is_pursued_goal_reached:
                self.goal_array[index_pursued_goal]["r|p"] += 1 - self.ema_speed
            after = self.goal_array[index_pursued_goal]["r|p"]
            self.goal_array[index_pursued_goal]["delta_r|p"] = \
                self.ema_speed * self.goal_array[index_pursued_goal]["delta_r|p"] + \
                (1 - self.ema_speed) * (after - before)
            # p(r|~p): for all goals but the current, increase it's freq if reached else decrease
            self.goal_array[:index_pursued_goal]["r|~p"] *= self.ema_speed
            self.goal_array[index_pursued_goal + 1:]["r|~p"] *= self.ema_speed
            all_goals_reached_indices = self.find(observed_goal)
            for index in all_goals_reached_indices:
                if index != index_pursued_goal:
                    self.goal_array[index]["r|~p"] += 1 - self.ema_speed
            # if no goals were reached and the library is not full, add the observed_goal as a new goal
            if self.n_goals < self.library_size and all_goals_reached_indices.shape[0] == 0:
                self.goal_array[self.n_goals] = np.array((observed_goal, 0, 0, 0), dtype=self.goal_dtype)
                if vision is not None:
                    with self.vision_array.get_lock():
                        self.vision_array[self.n_goals] = vision
                self._n_goals.nparray[()] += 1
        # return 1.0 if is_pursued_goal_reached else 0.0
        return self.n_goals < self.library_size and all_goals_reached_indices.shape[0] == 0

    def find(self, goal):
        return np.where(np_discretization_reward(self.registered_goals["intensities"], goal))[0]

    def index_of(self, goal):
        if self.n_goals == 0:
            return 0
        return np.argmax(np_discretization_reward(self.registered_goals["intensities"], goal) == 1)

    def select_goal_uniform(self):
        if self.n_goals == 1:
            return 0, self.goal_array[0]["intensities"]
        index = np.random.randint(1, self.n_goals)
        return index, np.copy(self.goal_array[index]["intensities"])

    def select_goal_uniform_learned_only(self, min_prob=0.0005):
        index_ok = (np.argwhere(self.registered_goals[1:]["r|p"] > min_prob) + 1).reshape((-1,))
        if len(index_ok) == 0:
            return 0, self.goal_array[0]["intensities"]
        choice = np.random.choice(index_ok)
        return choice, np.copy(self.goal_array[choice]["intensities"])

    def select_goal_uniform_reachable_only(self, min_prob=0.0005):
        index_ok = (np.argwhere(self.registered_goals[1:]["r|~p"] > min_prob) + 1).reshape((-1,))
        if len(index_ok) == 0:
            return 0, self.goal_array[0]["intensities"]
        choice = np.random.choice(index_ok)
        return choice, np.copy(self.goal_array[choice]["intensities"])

    def _get_learning_potential_probs(self, temperature=0.015, min_reach_prob=0.001):
        where = np.where(np.logical_or(
            self.registered_goals[1:]["r|~p"] > min_reach_prob,
            self.registered_goals[1:]["r|p"] > min_reach_prob))[0]
        reachable_goals = self.registered_goals[1:][where]
        if len(reachable_goals) == 0:
            selectable_goals = self.registered_goals[1:]
            where = np.arange(self.n_goals - 1)
        else:
            selectable_goals = reachable_goals
        logits = selectable_goals["delta_r|p"] / (1 - self.ema_speed)
        logits[np.where(logits < 0)] = 0
        p = softmax(logits / temperature)
        probs = np.zeros((self.n_goals,))
        probs[where + 1] = p
        probs /= np.sum(probs)
        return probs

    def select_goal_learning_potential(self, temperature=0.001, min_reach_prob=0.005):
        if self.n_goals == 1:
            return 0, self.goal_array[0]["intensities"]
        p = self._get_learning_potential_probs(temperature, min_reach_prob)
        goal_index = np.random.choice(self.n_goals, p=p)
        goal = self.goal_array[goal_index]["intensities"]
        return goal_index, np.copy(goal)

    def save_vision(self, path):
        os.mkdir(path)
        with open(path + "/library.txt", "w") as f:
            f.write(str(self))
        for i, data in enumerate(self.vision_array[1:self.n_goals]):
            data = data.reshape((data.shape[0], -1))
            png.from_array(data, "RGB").save(path + "/frame_{:04d}.png".format(i))

    def take_snapshot(self, iteration):
        self.snapshots[iteration] = np.copy(self.registered_goals)

    def dump(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def goal_info(self, goal_index):
        return self.goal_array[goal_index]

    def plot_per_goal(self, path, name):
        fig = plt.figure()
        iterations = list(sorted(self.snapshots))
        for goal_index in range(len(self.snapshots[iterations[-1]])):
            self.plot_goal(goal_index, fig)
            fig.savefig(path + "/{}_goal_{:04d}.png".format(name, goal_index))
            fig.clear()
        plt.close(fig)

    def plot_goal(self, goal_index, fig):
        ax_rp   = fig.add_subplot(421)
        ax_rnp  = fig.add_subplot(423)
        ax_drp  = fig.add_subplot(425)
        ax_diff = fig.add_subplot(427)
        ax_vision = fig.add_subplot(222)
        ax_tactile = fig.add_subplot(224)
        iterations = list(sorted(self.snapshots))
        current_iterations = []
        current_rp   = []
        current_rnp  = []
        current_drp  = []
        current_diff = []
        for i in iterations:
            if len(self.snapshots[i]) > goal_index:
                current_iterations.append(i)
                current_rp.append(self.snapshots[i][goal_index]["r|p"])
                current_rnp.append(self.snapshots[i][goal_index]["r|~p"])
                current_drp.append(self.snapshots[i][goal_index]["delta_r|p"] / (1 - self.ema_speed))
                current_diff.append(self.snapshots[i][goal_index]["r|p"] - self.snapshots[i][goal_index]["r|~p"])
        ax_rp.plot(current_iterations, current_rp)
        ax_rnp.plot(current_iterations, current_rnp)
        ax_drp.plot(current_iterations, current_drp)
        ax_diff.plot(current_iterations, current_diff)
        ax_drp.axhline(0, color="k", ls="--")
        ax_diff.axhline(0, color="k", ls="--")
        im_data = self.vision_array[goal_index]
        if im_data is not None:
            ax_vision.imshow(im_data)
        ax_tactile.plot(self.goal_array[goal_index]["intensities"])
        ax_rp.set_xticks([], [])
        ax_rnp.set_xticks([], [])
        ax_drp.set_xticks([], [])
        ax_vision.set_xticks([], [])
        ax_vision.set_yticks([], [])
        ax_tactile.set_yticks([], [])
        ax_tactile.set_ylim([0, 1])
        ax_rnp.set_ylim([0, 0.05])
        ax_rp.set_ylabel("r|p")
        ax_rnp.set_ylabel("r|~p")
        ax_drp.set_ylabel("delta r|p")
        ax_diff.set_ylabel("diff")

    def __str__(self):
        string = ""
        p = self._get_learning_potential_probs()
        for i, g in enumerate(self.registered_goals):
            learnt = "\033[91m " if g["r|~p"] + 1e-4 > g["r|p"] else "\033[92m "
            learnt += " r|p - r|~p  {:.4f} \033[0m ".format(g["r|p"] - g["r|~p"])
            string += "#{:4d}  r|p {: .4f}   r|~p {: .4f}   delta_r|p {: .4f}   prob: {: .4f}      {}\n".format(
                i, g["r|p"], g["r|~p"], g["delta_r|p"] / (1 - self.ema_speed), p[i], learnt)
        return string

    def print_with_header(self, header):
        print(header)
        print(self)

    def restore(self, path):
        with open(path, "rb") as f:
            other = pickle.load(f)
        assert(self.library_size == other.library_size)
        assert(self.goal_size == other.goal_size)
        assert(self.vision_shape == other.vision_shape)
        # assert(self.ema_speed == other.ema_speed)
        with self._n_goals.get_lock():
            self._n_goals.nparray[()] = other._n_goals.nparray[()]
        with self.goal_array.get_lock():
            self.goal_array[:] = other.goal_array[:]
        with self.vision_array.get_lock():
            self.vision_array[:] = other.vision_array[:]
        self.snapshots = other.snapshots


if __name__ == "__main__":
    import time

    n_threads = 8
    library_size = 500
    goal_size = 80
    vision_shape = (320, 240)
    ema_speed = 0.995
    goal_library_server = GoalLibrary(library_size, goal_size, vision_shape, ema_speed)
    N = 10000

    def target(goal_library, thread_id):
        print("thread_id{} started".format(thread_id))
        for i in range(N // n_threads):
            goal_index, goal = goal_library.select_goal_uniform_reachable_only()
            tactile = np.random.uniform(size=goal_size, low=0, high=0.4)
            ret = goal_library.register_goal(tactile, goal)
            if ret:
                print("thread_id{}".format(thread_id), "new goal!", goal_library.n_goals)


    print("Create processes ...")
    processes = [mp.Process(target=target, args=(goal_library_server, i)) for i in range(n_threads)]
    t0 = time.time()
    [p.start() for p in processes]
    [p.join() for p in processes]
    t1 = time.time()
    print("Registration per sec: {}".format(n_threads * (N // n_threads) / (t1 - t0)))
