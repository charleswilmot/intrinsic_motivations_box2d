import numpy as np


class Buffer:
    def __init__(self, shape, size):
        self.shape = shape
        self.size = size
        self._index = 0
        self.data = np.zeros(shape=[2, size] + list(shape), dtype=np.float32)

    def random_indices(self, size):
        return np.random.choice(self._index, size, replace=False)

    def incorporate(self, new_data_0, new_data_1):
        new_data_0 = np.array(new_data_0)
        new_data_1 = np.array(new_data_1)
        N = new_data_0.shape[0]
        if self._index == self.size:
            ind = self.random_indices(N)
            self.data[0, ind] = new_data_0
            self.data[1, ind] = new_data_1
        elif self._index + N > self.size:
            lim = self.size - self._index
            self.incorporate(new_data_0[:lim], new_data_1[:lim])
            self.incorporate(new_data_0[lim:], new_data_1[lim:])
        else:
            self.data[0, self._index: self._index + N] = new_data_0
            self.data[1, self._index: self._index + N] = new_data_1
            self._index += N

    def batch(self, size):
        return self.data[:, self.random_indices(size)]

    # def shuffle(self):
    #     np.random.shuffle(self.data[:self._index])

    # def batches(self, size):
    #     self.shuffle()
    #     return [self.data[i * size:(i + 1) * size] for i in range(int(np.ceil(self.size / size)))]


if __name__ == "__main__":
    b = Buffer((), 10)
    b.incorporate([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6])
    print(b.data)
    b.incorporate([7, 8, 9, 10, 11, 12], [7, 8, 9, 10, 11, 12])
    print(b.data)
