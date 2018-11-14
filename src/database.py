import numpy as np
import os
import tensorflow as tf
import time
import viewer


def parse_box2d_protobuf(example_proto):
    features = {
        'vision': tf.FixedLenFeature([], tf.string),
        'vision_shape': tf.FixedLenFeature([], tf.string),
        'positions': tf.FixedLenFeature([], tf.string),
        'speeds': tf.FixedLenFeature([], tf.string),
        'tactile_map': tf.FixedLenFeature([], tf.string),
        'actions_arr': tf.FixedLenFeature([], tf.string),
        'index': tf.FixedLenFeature([], tf.int64)
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    vision = tf.decode_raw(parsed_features["vision"], tf.uint8)
    vision_shape = tf.decode_raw(parsed_features["vision_shape"], tf.int32)
    positions = tf.decode_raw(parsed_features["positions"], tf.float32)
    speeds = tf.decode_raw(parsed_features["speeds"], tf.float32)
    tactile_map = tf.decode_raw(parsed_features["tactile_map"], tf.float32)
    actions_arr = tf.decode_raw(parsed_features["actions_arr"], tf.float32)
    index = tf.cast(parsed_features["index"], tf.int32)
    ret = {
        'vision': tf.reshape(vision, vision_shape),
        'positions': positions,
        'speeds': speeds,
        'tactile_map': tactile_map,
        'actions_arr': actions_arr,
        'index': index
    }
    return ret


class DatabaseDisplay:
    def __init__(self, path):
        filenames = [path + '/' + f for f in os.listdir(path)]
        self.dataset = tf.data.TFRecordDataset(filenames)
        self.dataset = self.dataset.map(map_func=parse_box2d_protobuf, num_parallel_calls=8)
        self.iterator = self.dataset.make_initializable_iterator()
        self.next = self.iterator.get_next()
        self.initilalizer = self.iterator.initializer

    def __call__(self, t=None, n=None):
        with tf.Session() as sess:
            stop = False
            start_time = time.time()
            win = viewer.Window()
            sess.run(self.initilalizer)
            try:
                while not stop:
                    ret = sess.run(self.next)
                    win.update(ret["vision"], ret["positions"], ret["speeds"], ret["tactile_map"])
                    n = n - 1 if n is not None else None
                    elapsed_time = time.time() - start_time if t is not None else None
                    stop = (n is not None and n <= 0) or (elapsed_time is not None and elapsed_time > t)
            except tf.errors.OutOfRangeError:
                pass


def _bytelist_feature(arr):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[arr.tobytes()]))


def _int64list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


class DatabaseWriter:
    def __init__(self, path, filename_pattern="chunk_{}.tfrecord", chunk_size=1e10):
        self.path = path
        os.mkdir(path)
        self.filename_pattern = filename_pattern
        self.chunk_size = chunk_size
        self._count = 0
        self._current_chunk_size = 0
        self._chunk_number = 0
        self._writer = self._get_writer()

    def _get_writer(self):
        return tf.python_io.TFRecordWriter(self.path + '/' + self.filename_pattern.format(self._chunk_number))

    def _write(self, string):
        if self._current_chunk_size > self.chunk_size:
            self._writer.close()
            self._chunk_number += 1
            self._current_chunk_size = 0
            self._writer = self._get_writer()
        self._writer.write(string)

    def _serialize(self, vision, positions, speeds, tactile_map, actions):
        actions_arr = np.array([actions[key] for key in sorted(actions)], dtype=np.float32)
        if len(actions_arr) != len(speeds):
            raise ValueError("Number of action to be stored in DB is incorrect")
        feature = {
            'vision': _bytelist_feature(vision.astype(np.uint8)),
            'vision_shape': _bytelist_feature(np.array(vision.shape).astype(np.int32)),
            'positions': _bytelist_feature(positions.astype(np.float32)),
            'speeds': _bytelist_feature(speeds.astype(np.float32)),
            'tactile_map': _bytelist_feature(tactile_map.astype(np.float32)),
            'actions_arr': _bytelist_feature(actions_arr.astype(np.float32)),
            'index': _int64list_feature([self._count])
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        self._count += 1
        ret = example.SerializeToString()
        self._current_chunk_size += len(ret)
        return ret

    def __call__(self, vision, positions, speeds, tactile_map, actions):
        string = self._serialize(vision, positions, speeds, tactile_map, actions)
        self._write(string)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._writer.close()


if __name__ == "__main__":
    path = "/tmp/box2d_2018_11_14_16_06_02/"
    display = DatabaseDisplay(path)
    display()
