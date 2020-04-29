import numpy as np
import time
from training import make_experiment_path
import os


MEM_PER_WORKER = 1890
MEM_BASELINE = 1000

class ClusterQueue:
    def __init__(self, **kwargs):
        kwargs["video"] = False
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
                       time_scale_factor=2, batch_size=10, buffer_size=10, goal_buffer_size=10, updates_per_episode=1,
                       env_step_length=1, **kwargs):
        super().__init__(video=video, n_workers=n_workers, n_trajectories=n_trajectories, save_every=save_every,
                         learning_rate=learning_rate, time_scale_factor=time_scale_factor, batch_size=batch_size,
                         buffer_size=buffer_size, goal_buffer_size=goal_buffer_size,
                         updates_per_episode=updates_per_episode, env_step_length=env_step_length, **kwargs)



description = "1_joint_2_levels_scaled_noise"
agency_conf_path = "../agencies/one_joint_2_levels.json"

cq = CheckpointQueue(
    description=description + "__checkpoint",
    json_model="../models/one_joint.json",
    skin_order="../models/one_joint_skin_order.pkl",
    agency_conf_path=agency_conf_path)

checkpoint_path = cq.experiment_path + "/checkpoints/000000100/"
print(checkpoint_path)
time.sleep(30)

for behaviour_noise_scale in [0.05, 0.1, 0.15, 0.2]:
    cq = ClusterQueue(
        n_workers=32,
        description=description + "__noise_{:.02f}".format(behaviour_noise_scale),
        actor_speed_ratio=100,
        train_actor_every=1,
        learning_rate=1e-3,
        discount_factor=0.7,
        behaviour_noise_scale=0.1,
        target_smoothing_noise_scale=0.075,
        n_trajectories=500000,
        save_every=500000,
        batch_size=64,
        time_scale_factor=10,
        buffer_size=128,
        goal_buffer_size=100,
        env_step_length=45,
        tau=0.01,
        restore_from=checkpoint_path,
        json_model="../models/one_joint.json",
        skin_order="../models/one_joint_skin_order.pkl",
        agency_conf_path=agency_conf_path)


################################################################


description = "1_joint_1_level_scaled_noise"
agency_conf_path = "../agencies/one_joint_agency.json"

cq = CheckpointQueue(
    description=description + "__checkpoint",
    json_model="../models/one_joint.json",
    skin_order="../models/one_joint_skin_order.pkl",
    agency_conf_path=agency_conf_path)

checkpoint_path = cq.experiment_path + "/checkpoints/000000100/"
print(checkpoint_path)
time.sleep(30)

for behaviour_noise_scale in [0.05, 0.1, 0.15, 0.2]:
    cq = ClusterQueue(
        n_workers=32,
        description=description + "__noise_{:.02f}".format(behaviour_noise_scale),
        actor_speed_ratio=100,
        train_actor_every=1,
        learning_rate=1e-3,
        discount_factor=0.7,
        behaviour_noise_scale=0.1,
        target_smoothing_noise_scale=0.075,
        n_trajectories=150000,
        save_every=150000,
        batch_size=64,
        time_scale_factor=10,
        buffer_size=128,
        goal_buffer_size=100,
        env_step_length=45,
        tau=0.01,
        restore_from=checkpoint_path,
        json_model="../models/one_joint.json",
        skin_order="../models/one_joint_skin_order.pkl",
        agency_conf_path=agency_conf_path)
