import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle


parser = argparse.ArgumentParser()
parser.add_argument(
    '-e', '--experiment',
    type=str,
    action='store',
    default=None,
    help="Path to an experiment."
)

parser.add_argument(
    '-a', '--array',
    type=str,
    action='store',
    default=None,
    help="Path to an array."
)

parser.add_argument(
    '-i', '--iteration',
    type=int,
    action='store',
    default=10000,
    help="Iteration."
)

parser.add_argument(
    '--save',
    action='store_true',
    help="If this flag is set, save the plots."
)

parser.add_argument(
    '--show',
    action='store_true',
    help="If this flag is set, show the plots."
)

## Task: plot number of contact per 1000 iteration, partitioned in self contact / ball contact


def count_contacts(iteration_data):
    n_self_contact, n_ball_contact = 0, 0
    for member in iteration_data:
        for other in iteration_data[member]:
            if "Arm" in other:
                n_self_contact += 1
            if "Ball" in other:
                n_ball_contact += 1
    n_self_contact /= 2  # (two time more self contacts ...)
    return n_self_contact, n_ball_contact


def process_one_worker_data(one_worker_data):
    indices = np.array(sorted(one_worker_data))
    n_self_contact, n_ball_contact = np.zeros_like(indices, dtype=np.uint32), np.zeros_like(indices, dtype=np.uint32)
    for i, j in enumerate(indices):
        n_self_contact[i], n_ball_contact[i] = count_contacts(one_worker_data[j])
    current_index = 0
    current_bound = 0
    WINDOW = 1000
    n_self_contact_windowed = []
    n_ball_contact_windowed = []
    while current_index < len(indices) - 1:
        next_index = current_index
        next_bound = current_bound + WINDOW
        while next_index < len(indices) and indices[next_index] < next_bound:
            next_index += 1
        if next_index < len(indices):
            n_self_contact_windowed.append(np.sum(n_self_contact[current_index:next_index]) / WINDOW)
            n_ball_contact_windowed.append(np.sum(n_ball_contact[current_index:next_index]) / WINDOW)
        current_index = next_index
        current_bound = next_bound
    return n_self_contact_windowed, n_ball_contact_windowed


def agregate_workers(process_data):
    min_length = min([len(data[0]) for data in process_data])
    self_arrays, ball_arrays = [], []
    for i, (self_list, ball_list) in enumerate(process_data):
        self_arrays.append(self_list[:min_length])
        ball_arrays.append(ball_list[:min_length])
    self_mean = np.nanmean(self_arrays, axis=0)
    ball_mean = np.nanmean(ball_arrays, axis=0)
    return self_mean, ball_mean


def process_data(data):
    processed_data = [process_one_worker_data(one_worker_data) for one_worker_data in data]
    return agregate_workers(processed_data)


def plot_contact_freq(path, iteration, title, show=True, save=True):
    data = get_experiment_data(path, iteration)
    self_mean, ball_mean = process_data(data)
    X = np.arange(len(self_mean)) * 1000
    plt.fill_between(X, self_mean, label="self contacts")
    plt.fill_between(X, self_mean, ball_mean + self_mean, label="ball contacts")
    plt.plot(X, self_mean, "k-")
    plt.plot(X, self_mean + ball_mean, "k-")
    plt.title(title)
    plt.legend()
    if show:
        plt.show()
    if save:
        plt.savefig(path + "/contacts_freq_{}.png".format(iteration))
    plt.clf()


def get_experiment_data(path, iteration):
    logpath = path + "/log/"
    workers_paths = [logpath + x for x in os.listdir(logpath)]
    data = []
    for p in workers_paths:
        with open(p + "/contacts_{}.pkl".format(iteration), "rb") as f:
            data.append(pickle.load(f))
    return data


args = parser.parse_args()
if args.experiment is None and args.array is None:
    print("Must provide a path to an experiment or to an array")


if args.experiment is not None:
    title = os.path.basename(args.experiment)
    plot_contact_freq(args.experiment, args.iteration, title, args.show, args.save)
elif args.array is not None:
    experiment_paths = [args.array + x for x in os.listdir(args.array) if os.path.isdir(args.array + x)]
    for p in experiment_paths:
        title = os.path.basename(p)
        print(title)
        plot_contact_freq(p, args.iteration, title, args.show, args.save)
