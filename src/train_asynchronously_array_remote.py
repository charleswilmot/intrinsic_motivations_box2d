#!/usr/bin/env python3
#SBATCH --partition sleuths
#SBATCH --cpus-per-task 32
#SBATCH --mem=32000


import os
import re


array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
array_path = "../experiments/array_no_entropy_reg/"
commands = []


for discount_factor, df_dir_name in zip([0, 0.5, 0.85, 0.95, 0.99], ["df_000", "df_050", "df_085", "df_095", "df_099"]):
    sub_path = array_path + "{}/".format(df_dir_name)

    for exp_type in ["minimize", "maximize", "target_0001", "target_0005", "target_0010", "target_0020", "target_0030", "target_0040", "target_0050", "target_0060"]:
        sub_sub_path = sub_path + exp_type + "/"

        if re.match("minimize", exp_type):
            commands.append("srun python3 train_asynchronously.py {} -df {} -nw 16 -np 2 --minimize-pred-err 0.042".format(sub_sub_path, discount_factor))

        if re.match("maximize", exp_type):
            commands.append("srun python3 train_asynchronously.py {} -df {} -nw 16 -np 2 --maximize-pred-err 0.042".format(sub_sub_path, discount_factor))

        if re.match("target_([0-9]+)", exp_type):
            m = re.match("target_([0-9]+)", exp_type)
            target = int(m.group(1)) / 1000
            commands.append("srun python3 train_asynchronously.py {} -df {} -nw 16 -np 2 --target-pred-err {}".format(sub_sub_path, discount_factor, target))


cmd = commands[array_id] if array_id < len(commands) else ":"
print("array ID = {}\t\t{}".format(array_id, cmd))
os.system(cmd)
