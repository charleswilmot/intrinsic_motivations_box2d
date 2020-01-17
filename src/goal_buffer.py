import numpy as np
from scipy.spatial import distance_matrix


class GoalBuffer:
    def __init__(self, buffer_size, goal_size):
        self._index_last = 0
        self._goal_size = goal_size
        self._buffer_size = buffer_size
        self._buffer = np.zeros((buffer_size, goal_size))
        self._ploting_data_buffer = [None] * buffer_size
        self._distance_matrix = np.zeros((buffer_size, buffer_size))

    def _maybe_register_one_goal(self, goal, ploting_data):
        distances = self._distance_to_buffer(goal)
        if self._index_last < self._buffer_size:
            self._place_goal(goal, ploting_data, self._index_last, distances)
            self._index_last += 1
        else:
            goal_min_distance = np.min(distances)
            buffer_min_distance, index = self._min_distance()
            if goal_min_distance > buffer_min_distance:
                self._place_goal(goal, ploting_data, index, distances)

    def _place_goal(self, goal, ploting_data, index, distances):
            self._buffer[index] = goal
            self._ploting_data_buffer[index] = ploting_data
            self._distance_matrix[index, :self._index_last] = distances
            self._distance_matrix[:self._index_last, index] = distances
            self._distance_matrix[index, index] = np.inf

    def _distance_to_buffer(self, goal):
        return distance_matrix(self._buffer[:self._index_last], np.array(goal).reshape((1, self._goal_size)))[:, 0]

    def _min_distance(self):
        index = np.unravel_index(
            np.argmin(self._distance_matrix[:self._index_last, :self._index_last]),
            (self._index_last, self._index_last)
        )
        return self._distance_matrix[index], index[0]

    def register_many(self, goals, ploting_datas):
        for goal, ploting_data in zip(goals, ploting_datas):
            self._maybe_register_one_goal(goal, ploting_data)

    def register_one(self, goal, ploting_data):
        self._maybe_register_one_goal(goal, ploting_data)

    def sample(self):
        index = np.random.randint(self._index_last)
        index_nearest = np.argmin(self._distance_matrix[index])
        epsilon = np.random.uniform()
        goal = self._buffer[index] * epsilon + (1 - epsilon) * self._buffer[index_nearest]
        ploting_data_0 = self._ploting_data_buffer[index]
        ploting_data_1 = self._ploting_data_buffer[index_nearest]
        return goal, (ploting_data_0, ploting_data_1)



if __name__ == "__main__":
    np.set_printoptions(precision=3)
    gb = GoalBuffer(500, 2)

    for i in range(20):
        gb.register(np.random.uniform(size=(2000, 2)))

    import matplotlib.pyplot as plt

    samples = np.array([gb.sample() for i in range(1000)])
    x = samples[:, 0]
    y = samples[:, 1]
    plt.scatter(x=x, y=y)

    x = gb._buffer[:, 0]
    y = gb._buffer[:, 1]
    plt.scatter(x=x, y=y)

    plt.show()
