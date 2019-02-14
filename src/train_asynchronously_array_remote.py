#!/usr/bin/env python3
#SBATCH --partition sleuths
#SBATCH --cpus-per-task 16
#SBATCH --mem=16000
#SBATCH --output=/dev/null


import os
import re
import time


array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
array_path = "../experiments/array_df_095_continous_vs_stages_vs_separate/"
slurm_log_path = "{}/log_array_id_{}.log".format(array_path, array_id)


def make_command(path, args_dict):
    command = "srun --output={} python3 train_asynchronously.py {}".format(slurm_log_path, path)
    for flag in args_dict:
        val = args_dict[flag]
        formated_val = ""
        if type(val) == list:
            for v in val:
                formated_val = formated_val + "{} ".format(v)
            formated_val = formated_val[:-1]
        else:
            formated_val = str(val)
        command += " {} {}".format(flag, formated_val)
    return command


# def make_command(path, df, reward_type, *reward_args, continuous=False, n_stages=4):
#     for arg in reward_args:
#         reward_type = reward_type + " {}".format(arg)
#     command = "srun python3 train_asynchronously.py {} -df {} -nw 16 -np 2 {} -s {}".format(path, df, reward_type, n_stages)
#     if continuous:
#         command = command + " -c"
#     return command


# def array_wrt_discount_factor():
#     commands = []
#     for discount_factor, df_dir_name in zip([0, 0.5, 0.85, 0.95, 0.99], ["df_000", "df_050", "df_085", "df_095", "df_099"]):
#         sub_path = array_path + "{}/".format(df_dir_name)
#
#         for exp_type in ["minimize", "maximize", "target_0001", "target_0005", "target_0010", "target_0020", "target_0030", "target_0040", "target_0050", "target_0060"]:
#             sub_sub_path = sub_path + exp_type + "/"
#
#             if re.match("minimize", exp_type):
#                 commands.append(make_command(sub_sub_path, discount_factor, "--minimize-pred-err", 0.042))
#             if re.match("maximize", exp_type):
#                 commands.append(make_command(sub_sub_path, discount_factor, "--maximize-pred-err", 0.042))
#             if re.match("target_([0-9]+)", exp_type):
#                 m = re.match("target_([0-9]+)", exp_type)
#                 target = int(m.group(1)) / 1000
#                 commands.append(make_command(sub_sub_path, discount_factor, "--target-pred-err", target))
#     return commands


def format_reward_type(arg):
    return re.match("--(.*)-pred-err", arg).group(1)


def format_reward_params(*reward_param):
    s = ""
    for p in reward_param:
        s = s + "{:.3f}_".format(p).replace(".", "")
    return s[:-1]


def array_wrt_training_type():
    commands, paths = [], []
    for continuous in [True, False]:
        format_continuous = "continuous" if continuous else "stages"
        for separate in [True, False]:
            format_separate = "separate" if separate else "joint"
            for reward_type, reward_param in [("--minimize-pred-err", [0.042]),
                                              ("--maximize-pred-err", [0.042]),
                                              ("--target-pred-err", [0.01]),
                                              ("--target-pred-err", [0.03]),
                                              ("--target-pred-err", [0.05]),
                                              ("--range-pred-err", [0.01, 0.03])]:
                path = array_path + "/{}_{}_{}_{}".format(
                    format_reward_type(reward_type),
                    format_reward_params(*reward_param),
                    format_continuous,
                    format_separate)
                args_dict = {}
                args_dict[reward_type] = reward_param
                args_dict["-df"] = 0.95
                args_dict["-nw"] = 16
                args_dict["-np"] = 2
                args_dict["-s"] = 4
                if separate:
                    args_dict["--separate"] = ""
                if continuous:
                    args_dict["--continuous"] = 12000
                commands.append(make_command(path, args_dict))
                paths.append(path)
    return commands, paths


commands, paths = array_wrt_training_type()
if array_id < len(commands):
    cmd = commands[array_id]
    experiment_path = paths[array_id]
    print("array ID = {}\t\t{}".format(array_id, cmd))
    # time.sleep(2 * (array_id % 30))
    os.system(cmd)
    os.rename(slurm_log_path, experiment_path + "/out.txt")
