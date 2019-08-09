import os
import png
import numpy as np
import discretization
from scipy.special import softmax
import pickle
import matplotlib.pyplot as plt


class GoalWarehouse:
    def __init__(self, warehouse_size, goal_size, ema_speed):
        self.dtgoal = np.dtype(
            [("intensities", np.float32, goal_size),
             ("r|p", np.float32),
             ("r|~p", np.float32),
             ("delta_r|p", np.float32)])
        self.goals = np.zeros(warehouse_size, dtype=self.dtgoal)
        self._vision_related_to_the_goals = [None] * warehouse_size
        self.goals[0] = np.array((np.zeros(goal_size), 1, 1, 0), dtype=self.dtgoal)
        self.ema_speed = ema_speed
        self.ema_speed_slow = self.ema_speed ** (1 / 4)
        self.snapshots = {}
        self._n_goals = 1
        self._warehouse_size = warehouse_size
        self._total_adds = 0

    def add_goals(self, goals, pursued):
        for goal in goals:
            self.add_goal(goal, pursued)

    def add_goal(self, tactile, current_goal, vision=None):
        tactile, current_goal = np.array(tactile, dtype=np.float32), np.array(current_goal, dtype=np.float32)
        self._total_adds += 1
        is_current_goal_reached = discretization.np_discretization_reward(tactile, current_goal)
        all_goals_reached_indices = self.find(tactile)
        index_current_goal = np.where((self.goals["intensities"] == current_goal).all(axis=1))[0][0]
        # p(r|p) and p(r|~p):
        # p(r|p): increase the freq of the current goal if reached, else decrease it
        # delta_r|p
        before = self.goals[index_current_goal]["r|p"]
        self.goals[index_current_goal]["r|p"] *= self.ema_speed
        if is_current_goal_reached:
            self.goals[index_current_goal]["r|p"] += 1 - self.ema_speed
        after = self.goals[index_current_goal]["r|p"]
        self.goals[index_current_goal]["delta_r|p"] = \
            self.ema_speed_slow * self.goals[index_current_goal]["delta_r|p"] + \
            (1 - self.ema_speed_slow) * (after - before)
        # p(r|~p): for all goals but the current, increase it's freq if reached else decrease
        self.goals[:index_current_goal]["r|~p"] *= self.ema_speed_slow
        self.goals[index_current_goal + 1:]["r|~p"] *= self.ema_speed_slow
        for index in all_goals_reached_indices:
            if index != index_current_goal:
                self.goals[index]["r|~p"] += 1 - self.ema_speed_slow
        # if no goals were reached and the warehouse is not full, add the tactile as a goal
        if self._n_goals < self._warehouse_size and all_goals_reached_indices.shape[0] == 0:
            self.goals[self._n_goals] = np.array(
                (tactile,
                 0, # 1 / self._total_adds,
                 0, # 1 / self._total_adds,
                 0), dtype=self.dtgoal)
            self._vision_related_to_the_goals[self._n_goals] = vision
            self._n_goals += 1
        return 1.0 if is_current_goal_reached else 0

    def find(self, goal):
        return np.where(discretization.np_discretization_reward(self.goals[:self._n_goals]["intensities"], goal))[0]

    def select_goal_uniform(self):
        if self._n_goals == 1:
            return self.goals[0]["intensities"]
        index = np.random.randint(1, self._n_goals)
        return self.goals[index]["intensities"]

    def _get_learning_potential_probs(self, temperature=0.015, min_reach_prob=0.001):
        where = np.where(np.logical_or(
            self.goals[1:self._n_goals]["r|~p"] > min_reach_prob,
            self.goals[1:self._n_goals]["r|p"] > min_reach_prob))[0]
        reachable_goals = self.goals[1:self._n_goals][where]
        if len(reachable_goals) == 0:
            selectable_goals = self.goals[1:self._n_goals]
            where = np.arange(self._n_goals - 1)
        else:
            selectable_goals = reachable_goals
        logits = selectable_goals["delta_r|p"] / (1 - self.ema_speed)
        logits[np.where(logits < 0)] = 0
        p = softmax(logits / temperature)
        probs = np.zeros((self._n_goals,))
        probs[where + 1] = p
        probs /= np.sum(probs)
        return probs

    def select_goal_learning_potential(self, temperature=0.001, min_reach_prob=0.005):
        if self._n_goals == 1:
            return self.goals[0]["intensities"]
        p = self._get_learning_potential_probs(temperature, min_reach_prob)
        return np.random.choice(self.goals[:self._n_goals], p=p)["intensities"]

    def save_vision(self, path):
        os.mkdir(path)
        with open(path + "/warehouse.txt", "w") as f:
            f.write(str(self))
        for i, data in enumerate(self._vision_related_to_the_goals[1:self._n_goals]):
            data = data.reshape((data.shape[0], -1))
            png.from_array(data, "RGB").save(path + "/frame_{:04d}.png".format(i))

    def take_snapshot(self, iteration):
        self.snapshots[iteration] = np.copy(self.goals[:self._n_goals])

    def dump(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def plot_per_goal(self, path, name):
        fig     = plt.figure()
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
        im_data = self._vision_related_to_the_goals[goal_index]
        if im_data is not None:
            ax_vision.imshow(im_data)
        ax_tactile.plot(self.goals[goal_index]["intensities"])
        ax_rp.set_xticks([], [])
        ax_rnp.set_xticks([], [])
        ax_drp.set_xticks([], [])
        ax_vision.set_xticks([], [])
        ax_vision.set_yticks([], [])
        ax_tactile.set_yticks([], [])
        ax_tactile.set_ylim([0, 1])

    def __str__(self):
        string = ""
        p = self._get_learning_potential_probs()
        for i, g in enumerate(self.goals[:self._n_goals]):
            learnt = "\033[91m " if g["r|~p"] + 1e-4 > g["r|p"] else "\033[92m "
            learnt += " r|p - r|~p  {:.4f} \033[0m ".format(g["r|p"] - g["r|~p"])
            string += "#{:4d}  r|p {: .4f}   r|~p {: .4f}   delta_r|p {: .4f}   prob: {: .4f}      {}\n".format(
                i, g["r|p"], g["r|~p"], g["delta_r|p"] / (1 - self.ema_speed), p[i], learnt)
        return string

    def print_with_header(self, header):
        print(header)
        print(self)


if __name__ == "__main__":
    warehouse_size = 45
    goal_size = 10
    ema_speed = 0.995
    gw = GoalWarehouse(warehouse_size, goal_size, ema_speed)

    for j in range(10):
        for i in range(100):
            goal = gw.select_goal_learning_potential(temperature=0.007)
            if goals_match(goal, np.zeros(goal_size)) and i != 0 and j < 4:
                tactile = np.zeros(goal_size)
            else:
                tactile = np.random.uniform(size=goal_size, low=0, high=0.4)
            gw.add_goal(tactile, goal)
        print(gw)









    # def add_goal_old2(self, tactile, current_goal):
    #     tactile, current_goal = np.array(tactile, dtype=np.float32), np.array(current_goal, dtype=np.float32)
    #     self._total_adds += 1
    #     is_current_goal_reached = goals_match(tactile, current_goal)
    #     all_goals_reached_indices = self.find(tactile)
    #     index_current_goal = np.where((self.goals["intensities"] == current_goal).all(axis=1))[0][0]
    #     # p(r) and p(~r): increase freq of all goals reached, decrease freq of all others
    #     self.goals["r"] *= self.ema_speed
    #     self.goals["~r"] *= self.ema_speed
    #     for index in range(self._n_goals):
    #         if index in all_goals_reached_indices:
    #             self.goals[index]["r"] += 1 - self.ema_speed
    #         else:
    #             self.goals[index]["~r"] += 1 - self.ema_speed
    #     # p(r|p) and p(r|~p):
    #     # p(r|p): increase the freq of the current goal if reached, else decrease it
    #     # delta_r|p
    #     before = self.goals[index_current_goal]["r|p"]
    #     self.goals[index_current_goal]["r|p"] *= self.ema_speed
    #     if is_current_goal_reached:
    #         self.goals[index_current_goal]["r|p"] += 1 - self.ema_speed
    #     after = self.goals[index_current_goal]["r|p"]
    #     self.goals[index_current_goal]["delta_r|p"] = \
    #         self.ema_speed * self.goals[index_current_goal]["delta_r|p"] + \
    #         (1 - self.ema_speed) * (after - before)
    #     # p(r|~p): for all goals but the current, increase it's freq if reached else decrease
    #     self.goals[:index_current_goal]["r|~p"] *= self.ema_speed
    #     self.goals[index_current_goal + 1:]["r|~p"] *= self.ema_speed
    #     for index in all_goals_reached_indices:
    #         if index != index_current_goal:
    #             self.goals[index]["r|~p"] += 1 - self.ema_speed
    #     # if no goals were reached and the warehouse is not full, add the tactile as a goal
    #     if self._n_goals < self._warehouse_size and all_goals_reached_indices.shape[0] == 0:
    #         self.goals[self._n_goals] = np.array(
    #             (tactile,
    #              1 / self._total_adds,
    #              1 - 1 / self._total_adds,
    #              1 / self._total_adds,
    #              1 / self._total_adds,
    #              0), dtype=self.dtgoal)
    #         self._n_goals += 1

    # def add_goal_old(self, tactile, current_goal):
    #     tactile, current_goal = np.array(tactile, dtype=np.float32), np.array(current_goal, dtype=np.float32)
    #     self._total_adds += 1
    #     is_current_goal_reached = goals_match(tactile, current_goal)
    #     all_goals_reached_indices = self.find(tactile)
    #     index_current_goal = np.where((self.goals["intensities"] == current_goal).all(axis=1))[0][0]
    #     # print(tactile, all_goals_reached_indices)
    #     # ALL: increase freq of all goals reached, decrease freq of all others
    #     self.goals["all"] *= self.ema_speed
    #     for index in all_goals_reached_indices:
    #         self.goals[index]["all"] += 1 - self.ema_speed
    #     # PURSUED: increase the freq of the current goal if reached, else decrease it
    #     self.goals[index_current_goal]["pursued"] *= self.ema_speed
    #     if is_current_goal_reached:
    #         self.goals[index_current_goal]["pursued"] += 1 - self.ema_speed
    #     # NOT PURSUED: for all goals but the current, increase it's freq if reached else decrease
    #     self.goals[:index_current_goal]["not_pursued"] *= self.ema_speed
    #     self.goals[index_current_goal + 1:]["not_pursued"] *= self.ema_speed
    #     for index in all_goals_reached_indices:
    #         if index != index_current_goal:
    #             self.goals[index]["not_pursued"] += 1 - self.ema_speed
    #     # if no goals were reached and the warehouse is not full, add the tactile as a goal
    #     if self._n_goals < self._warehouse_size and all_goals_reached_indices.shape[0] == 0:
    #         self.goals[self._n_goals] = np.array(
    #             (tactile,
    #              1 / self._total_adds,
    #              1 / self._total_adds,
    #              1 / self._total_adds), dtype=self.dtgoal)
    #         self._n_goals += 1
