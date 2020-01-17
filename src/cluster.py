import numpy as np
import time
from training import make_experiment_path
import os


MEM_PER_WORKER = 1890
MEM_BASELINE = 1000

class ClusterQueue:
    def __init__(self, **kwargs):
        self.experiment_path = make_experiment_path(
            lr=kwargs["learning_rate"] if "learning_rate" in kwargs else None,
            discount_factor=kwargs["discount_factor"] if "discount_factor" in kwargs else None,
            description=kwargs["description"])
        os.mkdir(self.experiment_path)
        os.mkdir(self.experiment_path + "/log")
        self.cmd_slurm = "sbatch --output {}/log/%N_%j.log".format(self.experiment_path)
        if "description" in kwargs:
            self.cmd_slurm += " --job-name {}".format(kwargs["description"])
        n_workers = kwargs["n_workers"] if "n_workers" in kwargs else 4
        self.cmd_slurm += " --mincpus {}".format(n_workers)
        self.cmd_slurm += " --mem {}".format(n_workers * MEM_PER_WORKER + MEM_BASELINE)
        self.cmd_slurm += " cluster.sh"
        self.cmd_python = ""
        for k, v in kwargs.items():
            flag = self._key_to_flag(k)
            arg = self._to_arg(flag, v)
            self.cmd_python += arg
        self.cmd_python += self._to_arg("--experiment-path", self.experiment_path)
        self.cmd = self.cmd_slurm + self.cmd_python
        print("\n" * 2, self.cmd_slurm)
        print(self.cmd_python, "\n" * 2)
        os.system(self.cmd)
        time.sleep(1)

    def _key_to_flag(self, key):
        return "--" + key.replace("_", "-")

    def _to_arg(self, flag, v):
        return " {} {}".format(flag, v)

    def watch_tail(self):
        os.system("watch tail -n 40 \"{}\"".format(self.experiment_path + "/log/*.log"))


# cq = ClusterQueue(
#     description="checkpoint",
#     video=False,
#     discount_factor=0.0001,
#     n_workers=1,
#     n_trajectories=100,
#     learning_rate=0,
#     batch_size=256,
#     buffer_size=1024,
#     behaviour_noise_scale=0.025,
#     target_smoothing_noise_scale=0.005,
#     goal_buffer_size=1000,
#     tau=0.05,
#     save_every=100,
#     updates_per_episode=1)


actor_speed_ratio = 30
train_actor_every = 5
learning_rate = 5e-4

for behaviour_noise_scale in [0.005, 0.01, 0.05, 0.1]:
    for target_smoothing_noise_scale in [0.005, 0.01, 0.05, 0.1]:
        cq = ClusterQueue(
            description="asr_{}_tae_{}_bns_{}_tsn_{}".format(actor_speed_ratio, train_actor_every, behaviour_noise_scale, target_smoothing_noise_scale),
            video=False,
            actor_speed_ratio=actor_speed_ratio,
            train_actor_every=train_actor_every,
            discount_factor=0.6,
            n_workers=32,
            n_trajectories=500000,
            save_every=100000,
            learning_rate=learning_rate,
            sequence_length=256,
            batch_size=256,
            buffer_size=1024,
            behaviour_noise_scale=behaviour_noise_scale,
            target_smoothing_noise_scale=target_smoothing_noise_scale,
            goal_buffer_size=1000,
            tau=0.05,
            updates_per_episode=1,
            restore_from="../experiments/2020_01_10-12.09.24_lr1.00e-04_discount_factor1.00e-04__checkpoint/checkpoints/000000100/")


# for learning_rate in [1e-4, 1e-3, 5e-3]:
#     for actor_speed_ratio in [10000, 1000, 100, 50, 10]:
#         train_actor_every = 1
#         cq = ClusterQueue(
#             description="bn_everywhere_but_leafs_asr_{}_tae_{}".format(actor_speed_ratio, train_actor_every),
#             video=False,
#             actor_speed_ratio=actor_speed_ratio,
#             train_actor_every=train_actor_every,
#             discount_factor=0.6,
#             n_workers=4,
#             n_trajectories=200000,
#             save_every=100000,
#             learning_rate=learning_rate,
#             sequence_length=256,
#             batch_size=256,
#             buffer_size=1024,
#             behaviour_noise_scale=0.025,
#             target_smoothing_noise_scale=0.005,
#             goal_buffer_size=1000,
#             tau=0.05,
#             updates_per_episode=1,
#             restore_from="../experiments/2020_01_10-12.09.24_lr1.00e-04_discount_factor1.00e-04__checkpoint/checkpoints/000000100/")
#
#
#     for train_actor_every in [10000, 1000, 100, 50, 10]:
#         actor_speed_ratio = 1
#         cq = ClusterQueue(
#             description="bn_everywhere_but_leafs_asr_{}_tae_{}".format(actor_speed_ratio, train_actor_every),
#             video=False,
#             actor_speed_ratio=actor_speed_ratio,
#             train_actor_every=train_actor_every,
#             discount_factor=0.6,
#             n_workers=4,
#             n_trajectories=200000,
#             save_every=100000,
#             learning_rate=learning_rate,
#             sequence_length=256,
#             batch_size=256,
#             buffer_size=1024,
#             behaviour_noise_scale=0.025,
#             target_smoothing_noise_scale=0.005,
#             goal_buffer_size=1000,
#             tau=0.05,
#             updates_per_episode=1,
#             restore_from="../experiments/2020_01_10-12.09.24_lr1.00e-04_discount_factor1.00e-04__checkpoint/checkpoints/000000100/")
