import numpy as np
import time
from training import make_experiment_path
import os


MEM_PER_WORKER = 900
MEM_BASELINE = 1000

class ClusterQueue:
    def __init__(self, **kwargs):
        self.experiment_path = make_experiment_path(
            clr=kwargs["critic_learning_rate"] if "critic_learning_rate" in kwargs else None,
            epsilon=kwargs["epsilon_init"] if "epsilon_init" in kwargs else None,
            discount_factor=kwargs["discount_factor"] if "discount_factor" in kwargs else None,
            description=kwargs["description"])
        os.mkdir(self.experiment_path)
        os.mkdir(self.experiment_path + "/log")
        self.cmd_slurm = "sbatch --output {}/log/%N_%j.log".format(self.experiment_path)
        if "description" in kwargs:
            self.cmd_slurm += " --job-name {}".format(kwargs["description"])
        n_workers = kwargs["n_workers"] if "n_workers" in kwargs else 4
        self.cmd_slurm += " --mincpus {}".format(int(n_workers / 2))
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


cq = ClusterQueue(
    description="concat_1e-4_4nets_her_first_batch_10_updates_10_big_net",
    discount_factor=0.5,
    n_workers=64,
    sequence_length=50,
    n_actions=11,
    flush_every=2000,
    n_trajectories=2000000,
    critic_learning_rate=1e-4,
    her_strategy="first",
    updates_per_episode=10,
    batch_size=10,
    skin_resolution=10,
    parametrization_type="concat")
