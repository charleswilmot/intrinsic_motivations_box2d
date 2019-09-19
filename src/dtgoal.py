import numpy as np
from scipy.special import softmax


def goals_match(goals0, goals1):
    return np.sum(np.abs(goals0 - goals1), axis=-1) < 0.9


class GoalLibrary:
    def __init__(self, library_size, goal_size, ema_speed):
        self.dtgoal = np.dtype(
            [("intensities", np.float32, goal_size),
             ("r|p", np.float32),
             ("r|~p", np.float32),
             ("delta_r|p", np.float32)])
        self.goals = np.zeros(library_size, dtype=self.dtgoal)
        self.goals[0] = np.array((np.zeros(goal_size), 1, 1, 0), dtype=self.dtgoal)
        self.ema_speed = ema_speed
        self._n_goals = 1
        self._library_size = library_size
        self._total_adds = 0

    def add_goals(self, goals, pursued):
        for goal in goals:
            self.add_goal(goal, pursued)

    def add_goal(self, tactile, current_goal):
        tactile, current_goal = np.array(tactile, dtype=np.float32), np.array(current_goal, dtype=np.float32)
        self._total_adds += 1
        is_current_goal_reached = goals_match(tactile, current_goal)
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
            self.ema_speed * self.goals[index_current_goal]["delta_r|p"] + \
            (1 - self.ema_speed) * (after - before)
        # p(r|~p): for all goals but the current, increase it's freq if reached else decrease
        self.goals[:index_current_goal]["r|~p"] *= self.ema_speed
        self.goals[index_current_goal + 1:]["r|~p"] *= self.ema_speed
        for index in all_goals_reached_indices:
            if index != index_current_goal:
                self.goals[index]["r|~p"] += 1 - self.ema_speed
        # if no goals were reached and the library is not full, add the tactile as a goal
        if self._n_goals < self._library_size and all_goals_reached_indices.shape[0] == 0:
            self.goals[self._n_goals] = np.array(
                (tactile,
                 1 / self._total_adds,
                 1 / self._total_adds,
                 0), dtype=self.dtgoal)
            self._n_goals += 1
        return 1.0 if is_current_goal_reached else 0

    def find(self, goal):
        return np.where(goals_match(self.goals[:self._n_goals]["intensities"], goal))[0]

    def select_goal_uniform(self):
        index = np.random.randint(self._n_goals)
        return self.goals[index]["intensities"]

    def select_goal_learning_potential(self, temperature=1):
        return np.random.choice(self.goals[:self._n_goals], p=softmax(self.goals[:self._n_goals]["delta_r|p"] / (1 - self.ema_speed) / temperature))["intensities"]


    def __str__(self):
        string = ""
        temperature = 0.007
        probs = softmax(self.goals[:self._n_goals]["delta_r|p"] / (1 - self.ema_speed) / temperature)
        for i, g in enumerate(self.goals[:self._n_goals]):
            string += "#{}  r|p {:.4f}   r|~p {:.4f}   delta_r|p {:.4f}   prob: {:.4f}\n".format(
                i, g["r|p"], g["r|~p"], g["delta_r|p"] / (1 - self.ema_speed), probs[i])
        return string

if __name__ == "__main__":
    library_size = 45
    goal_size = 10
    ema_speed = 0.995
    gw = GoalLibrary(library_size, goal_size, ema_speed)

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
    #     # if no goals were reached and the library is not full, add the tactile as a goal
    #     if self._n_goals < self._library_size and all_goals_reached_indices.shape[0] == 0:
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
    #     # if no goals were reached and the library is not full, add the tactile as a goal
    #     if self._n_goals < self._library_size and all_goals_reached_indices.shape[0] == 0:
    #         self.goals[self._n_goals] = np.array(
    #             (tactile,
    #              1 / self._total_adds,
    #              1 / self._total_adds,
    #              1 / self._total_adds), dtype=self.dtgoal)
    #         self._n_goals += 1
