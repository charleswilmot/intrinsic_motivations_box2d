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


class CheckpointQueue(ClusterQueue):
    def __init__(self, video=False, n_workers=1, n_trajectories=100, save_every=100, learning_rate=0.0,
                       sequence_length=10, batch_size=10, buffer_size=10, goal_buffer_size=10, updates_per_episode=1,
                       env_step_length=1, **kwargs):
        super().__init__(video=video, n_workers=n_workers, n_trajectories=n_trajectories, save_every=save_every,
                         learning_rate=learning_rate, sequence_length=sequence_length, batch_size=batch_size,
                         buffer_size=buffer_size, goal_buffer_size=goal_buffer_size,
                         updates_per_episode=updates_per_episode, env_step_length=env_step_length, **kwargs)



agency_conf_path = "../agencies/one_joint_2_levels_latent_action_size_2.txt"

cq = CheckpointQueue(
    description="one_joint_2_levels_checkpoint",
    json_model="../models/one_joint.json",
    skin_order="../models/one_joint_skin_order.pkl",
    agency_conf_path=agency_conf_path)

# name = "one_joint_2_levels_slowdown_64_latent_action_2"

# cq = ClusterQueue(
#     n_workers=32,
#     description=name + "_critic_checkpoint",
#     actor_speed_ratio=100000,
#     train_actor_every=100000,
#     learning_rate=5e-3,
#     discount_factor=0.7,
#     behaviour_noise_scale=0.15,
#     target_smoothing_noise_scale=0.0,
#     n_trajectories=10000,
#     save_every=10000,
#     batch_size=512,
#     sequence_length=32,
#     buffer_size=1024,
#     goal_buffer_size=100,
#     env_step_length=45,
#     json_model="../models/one_joint.json",
#     skin_order="../models/one_joint_skin_order.pkl",
#     agency_conf_path=agency_conf_path)



cq = ClusterQueue(
    n_workers=32,
    description="one_joint_2_levels_no_speed_diff_no_state_learning_asr_64_la_2_raw_distance",
    actor_speed_ratio=64,
    train_actor_every=2,
    learning_rate=5e-3,
    discount_factor=0.7,
    behaviour_noise_scale=0.15,
    target_smoothing_noise_scale=0.0,
    n_trajectories=50000,
    save_every=50000,
    batch_size=512,
    sequence_length=32,
    buffer_size=1024,
    goal_buffer_size=100,
    env_step_length=45,
    restore_from="../experiments/2020_02_07-11.17.38_lr1.00e-04_discount_factor9.00e-01__one_joint_2_levels_checkpoint/checkpoints/000000100/",
    json_model="../models/one_joint.json",
    skin_order="../models/one_joint_skin_order.pkl",
    agency_conf_path=agency_conf_path)



# for actor_speed_ratio in [1, 5]:
#     for train_actor_every in [10, 20, 40, 60, 100]:
#         cq = ClusterQueue(
#             n_workers=32,
#             description="one_joint_2_levels_shallow_asr{}_tae{}_goal_is_gstate_95percent".format(actor_speed_ratio, train_actor_every),
#             actor_speed_ratio=1,
#             train_actor_every=train_actor_every,
#             learning_rate=5e-3,
#             discount_factor=0.5,
#             behaviour_noise_scale=0.01,
#             target_smoothing_noise_scale=0.0,
#             n_trajectories=50000,
#             save_every=50000,
#             batch_size=512,
#             sequence_length=32,
#             buffer_size=1024,
#             goal_buffer_size=100,
#             env_step_length=45,
#             restore_from="../experiments/2020_02_04-13.37.21_lr5.00e-03_discount_factor5.00e-01__one_joint_2_levels_shallow_critic_checkpoint/checkpoints/000010000/",
#             json_model="../models/one_joint.json",
#             skin_order="../models/one_joint_skin_order.pkl",
#             agency_conf_path="../agencies/one_joint_2_levels_latent_action_size_1.txt")
