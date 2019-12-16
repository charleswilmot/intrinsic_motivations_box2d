import numpy as np
from scipy.spatial import distance_matrix


class GoalBuffer:
    def __init__(self, buffer_size, goal_size, keep_percent=0.1):
        self._index_last = 0
        self._current_fifo_index = -1
        self._goal_size = goal_size
        self._buffer_size = buffer_size
        self._buffer = np.zeros((buffer_size, goal_size))
        self._keep_percent = keep_percent

    def force_register(self, goals):
        self._append_fifo(goals)

    def register(self, goals):
        goals = np.array(goals)
        n_goals = len(goals)
        minimums = self._compute_min_distance_to_buffer(goals)
        argsort = np.argsort(minimums)
        keep = [goals[i] for i in argsort[-int(n_goals * self._keep_percent):]]
        self._append_fifo(keep)

    def _compute_min_distance_to_buffer(self, goals):
        return np.min(distance_matrix(self._buffer, goals), axis=0)

    def _append_fifo(self, keep):
        for k in keep:
            self._current_fifo_index += 1
            self._current_fifo_index %= self._buffer_size
            self._buffer[self._current_fifo_index] = k
            if self._index_last < self._buffer_size:
                self._index_last += 1

    def sample(self):
        return self._buffer[np.random.randint(self._index_last)]



if __name__ == "__main__":
    gb = GoalBuffer(10, 5)

    for i in range(10):
        gb.register(np.random.uniform(size=(20, 5)))
        print(gb.sample())
