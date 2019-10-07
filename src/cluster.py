import numpy as np
import time
from training import make_experiment_path
import os


MEM_PER_WORKER = 12200

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
        n_workers = kwargs["n_workers"] if "n_workers" in kwargs else 5
        self.cmd_slurm += " --mincpus {}".format(n_workers)
        self.cmd_slurm += " --mem {}".format(n_workers * MEM_PER_WORKER)
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
        time.sleep(10)

    def _key_to_flag(self, key):
        return "--" + key.replace("_", "-")

    def _to_arg(self, flag, v):
        return " {} {}".format(flag, v)

    def watch_tail(self):
        os.system("watch tail -n 40 \"{}\"".format(self.experiment_path + "/log/*.log"))


cq = ClusterQueue(description="reproduce_after_changes")
cq = ClusterQueue(description="reproduce_after_changes", discount_factor=0.5)
cq = ClusterQueue(description="reproduce_after_changes", discount_factor=0.95)
cq = ClusterQueue(description="reproduce_after_changes", discount_factor=0.99)
