import numpy as np
import os
import tensorflow as tf
import time
import re


tf_type = {
    'vision': tf.uint8,
    'positions': tf.float32,
    'speeds': tf.float32,
    'tactile_map': tf.float32,
    'actions': tf.float32
}


def parse_protobuf(key, shape=None):
    def parse(example_proto):
        features = {key: tf.FixedLenFeature([], tf.string)}
        parsed_features = tf.parse_single_example(example_proto, features)
        data = tf.decode_raw(parsed_features[key], tf_type[key])
        if key == 'vision':
            data = tf.reshape(data, shape)
        return {key: data}
    return parse


def dict_union(*args):
    ret = {}
    for d in args:
        ret.update(d)
    return ret


def add_index_key(index):
    return {"index": index}


def get_dataset(path, **kwargs):
    regex = r'.*chunk([0-9]+)\.tfr'
    filename = [x for x in os.listdir(path + '/positions') if re.match(regex, x) is not None][0]
    regex = r"sf[0-9]+.?[0-9]*_re[0-9]+_ae[0-9]+_n([0-9]+)_chunk[0-9]+.tfr"
    n_records = int(re.match(regex, filename).group(1))
    datasets = [tf.data.Dataset.range(n_records).map(add_index_key)]
    regex = r'.*chunk([0-9]+)\.tfr'
    keys = [key for key in kwargs if kwargs[key]]
    shape = np.load(path + '/vision/shape.npz') if 'vision' in keys else None
    for key in keys:
        filenames = [x for x in os.listdir(path + '/' + key) if re.match(regex, x) is not None]
        filenames.sort(key=lambda n: int(re.match(regex, n).group(1)))
        filepaths = [path + '/' + key + '/' + f for f in filenames]
        dataset = tf.data.TFRecordDataset(filepaths)
        dataset = dataset.map(map_func=parse_protobuf(key, shape), num_parallel_calls=8)
        datasets.append(dataset)
    dataset = tf.data.Dataset.zip(tuple(datasets))
    dataset = dataset.map(dict_union)
    return dataset


def _bytelist_feature(arr):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[arr.tobytes()]))


class DatabaseWriter:
    def __init__(self, path, simulation_freq, record_every, action_every, n_record, chunk_size=1e10):
        self.path = path
        self.chunk_size = chunk_size
        self._filename_pattern = \
            "sf{}_re{}_ae{}_n{}_".format(simulation_freq, record_every, action_every, n_record) + "chunk{}.tfr"
        self._keys = ["vision", "positions", "speeds", "actions", "tactile_map"]
        self._current_chunk_size = {key: 0 for key in self._keys}
        self._chunk_number = {key: 0 for key in self._keys}
        os.mkdir(path)
        for key in self._keys:
            os.mkdir(path + "/" + key)
        self._writers = {key: self._get_writer(key) for key in self._keys}
        self._features = {
            'vision': lambda data: _bytelist_feature(data.astype(np.uint8)),
            'positions': lambda data: _bytelist_feature(data.astype(np.float32)),
            'speeds': lambda data: _bytelist_feature(data.astype(np.float32)),
            'tactile_map': lambda data: _bytelist_feature(data.astype(np.float32)),
            'actions': lambda data: _bytelist_feature(data.astype(np.float32))
        }
        self._write_shape_to_disk = True

    def _get_writer(self, key):
        path = self.path + '/' + key + '/' + self._filename_pattern.format(self._chunk_number[key])
        return tf.python_io.TFRecordWriter(path)

    def _write(self, key, string):
        if self._current_chunk_size[key] > self.chunk_size:
            self._writers[key].close()
            self._chunk_number[key] += 1
            self._current_chunk_size[key] = 0
            self._writers[key] = self._get_writer(key)
        self._writers[key].write(string)
        self._current_chunk_size[key] += len(string)

    def _serialize(self, key, data):
        feature = {key: self._features[key](data)}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example.SerializeToString()

    def __call__(self, **kwargs):
        if self._write_shape_to_disk:
            self._write_shape_to_disk = False
            with open(self.path + '/vision/shape.npz', "wb") as f:
                np.save(f, np.array(kwargs["vision"].shape, dtype=np.int32))
        for key in self._keys:
            self._write(key, self._serialize(key, kwargs[key]))
        # store vision shape in a text file or so...

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        for key in self._writers:
            self._writers[key].close()
