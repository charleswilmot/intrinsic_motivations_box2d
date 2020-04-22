import numpy as np
from copy import deepcopy


class Buffer:
    def __init__(self, size):
        self._size = size
        self._current_insert_index = 0
        self._buffer_is_built = False

    def incorporate(self, transition_0, transition_1):
        if not self._buffer_is_built:
            self.build_buffer(transition_0)
        index = self.get_insertion_index()
        self.incorporate_at(index, transition_0, transition_1)

    def build_buffer(self, transition):
        dtype_description = []
        dtype_description += [(key, np.float32, val.shape) for key, val in transition.items()]
        dtype_description += [(key.replace("0", "1"), np.float32, val.shape) for key, val in transition.items()]
        self._dtype = np.dtype(dtype_description)
        self._buffer = np.zeros(self._size, dtype=self._dtype)
        self._buffer_is_built = True

    def get_insertion_index(self):
        if self._current_insert_index < self._size:
            self._current_insert_index += 1
            return self._current_insert_index - 1
        else:
            return np.random.randint(0, self._size)

    def incorporate_at(self, index, transition_0, transition_1):
        for key_0, val_0 in transition_0.items():
            self._buffer[index][key_0] = val_0
        for key_1, val_1 in transition_1.items():
            self._buffer[index][key_1.replace("0", "1")] = val_1

    def batch(self, size):
        if size >= self._current_insert_index:
            return self._buffer[:self._current_insert_index]
        else:
            index_start = np.random.randint(0, self._current_insert_index - size)
            return self._buffer[index_start:index_start + size]
