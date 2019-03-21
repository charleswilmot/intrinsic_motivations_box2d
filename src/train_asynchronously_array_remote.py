#!/usr/bin/env python3
#SBATCH --partition sleuths
#SBATCH --cpus-per-task 16
#SBATCH --mem=32000
#SBATCH --output=/dev/null


import os
import re
import time
import numpy as np

try:
    array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
except:
    array_id = -1
array_path = "../experiments/discrete_a3c/standard_experiments_alr_5e-5_clr_5e-5_mlr_1e-4_reg_5e-3_sl_32/"
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
        if p != "":
            s = s + "{:.3f}_".format(p).replace(".", "")
    return s[:-1]


def format_discount_factor(df):
    return "{:03d}".format(int(df * 1000) // 10)


def format_learning_rate(lr):
    return "{:.2e}".format(lr)


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


def array_wrt_learning_rates():
    commands, paths = [], []
    for log_clr in np.linspace(-4, -2, 5):
        clr = 10 ** log_clr
        for log_alr in np.linspace(-6, -4, 5):
            alr = 10 ** log_alr
            path = array_path + "/clr_10e{:3f}_alr_10e{:3f}".format(log_clr, log_alr)
            args_dict = {}
            args_dict["--target-pred-err"] = [0.035]
            args_dict["-df"] = 0.85
            args_dict["-nw"] = 32
            args_dict["-np"] = 2
            args_dict["-sl"] = 256
            args_dict["--continuous"] = 5000
            args_dict["-clr"] = clr
            args_dict["-alr"] = alr
            args_dict["-mlr"] = 1e-5
            commands.append(make_command(path, args_dict))
            paths.append(path)
    return commands, path


def array_wrt_model_learning_rates():
    commands, paths = [], []
    for log_mlr in np.linspace(-7, -2, 11):
        mlr = 10 ** log_mlr
        path = array_path + "/mlr_1e{:3f}".format(log_mlr)
        args_dict = {}
        args_dict["--target-pred-err"] = [0.035]
        args_dict["-df"] = 0.85
        args_dict["-nw"] = 32
        args_dict["-np"] = 2
        args_dict["-sl"] = 256
        args_dict["--continuous"] = 5000
        args_dict["-clr"] = 1e-3
        args_dict["-alr"] = 1e-5
        args_dict["-mlr"] = mlr
        commands.append(make_command(path, args_dict))
        paths.append(path)
    return commands, path


def array_test_targets():
    commands, paths = [], []
    for reward_type, reward_param in [("--target-pred-err", [0.010]),
                                      ("--target-pred-err", [0.015]),
                                      ("--target-pred-err", [0.020]),
                                      ("--target-pred-err", [0.025]),
                                      ("--target-pred-err", [0.030]),
                                      ("--target-pred-err", [0.035]),
                                      ("--target-pred-err", [0.040]),
                                      ("--target-pred-err", [0.045]),
                                      ("--target-pred-err", [0.050])]:
        path = array_path + "/{}_{}".format(
            format_reward_type(reward_type),
            format_reward_params(*reward_param))
        args_dict = {}
        args_dict[reward_type] = reward_param
        args_dict["-df"] = 0.85
        args_dict["-nw"] = 32
        args_dict["-np"] = 2
        args_dict["--continuous"] = 5000
        args_dict["-sl"] = 256
        args_dict["-clr"] = 1e-3
        args_dict["-mlr"] = 1e-5
        args_dict["-alr"] = 1e-5
        commands.append(make_command(path, args_dict))
        paths.append(path)
    return commands, paths


def array_standard_experiments():
    commands, paths = [], []
    for df in [0, 0.5, 0.85]:
        for reward_type, reward_param in [("--minimize-pred-err", [0.042]),
                                          ("--target-pred-err", [0.015]),
                                          ("--target-pred-err", [0.035])]:
            path = array_path + "/{}_{}_{}".format(
                format_reward_type(reward_type),
                format_reward_params(*reward_param),
                format_discount_factor(df))
            args_dict = {}
            args_dict[reward_type] = reward_param
            args_dict["-df"] = df
            args_dict["-nw"] = 32
            args_dict["-np"] = 2
            args_dict["--continuous"] = 5000
            args_dict["-sl"] = 256
            args_dict["-clr"] = 1e-3
            args_dict["-mlr"] = 1e-5
            args_dict["-alr"] = 1e-5
            commands.append(make_command(path, args_dict))
            paths.append(path)
    return commands, paths


def array_wrt_buffer_size():
    commands, paths = [], []
    for mbs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        path = array_path + "/{}".format(mbs)
        args_dict = {}
        args_dict["--target-pred-err"] = [0.035]
        args_dict["-df"] = 0.85
        args_dict["-nw"] = 32
        args_dict["-np"] = 2
        args_dict["--continuous"] = 5000
        args_dict["-sl"] = 256
        args_dict["-clr"] = 1e-3
        args_dict["-mlr"] = 1e-5
        args_dict["-alr"] = 1e-5
        args_dict["-mbs"] = mbs
        commands.append(make_command(path, args_dict))
        paths.append(path)
    return commands, paths


def array_standard_experiments_discrete_a3c():
    commands, paths = [], []
    for df in [0, 0.5, 0.85, 0.99]:
        for reward_type, reward_param in [("--minimize-pred-err", [""]),
                                          ("--maximize-pred-err", [""]),
                                          ("--target-pred-err", [0.005]),
                                          ("--target-pred-err", [0.010]),
                                          ("--target-pred-err", [0.015])]:
            path = array_path + "/{}_{}_{}".format(
                format_reward_type(reward_type),
                format_reward_params(*reward_param),
                format_discount_factor(df))
            args_dict = {}
            args_dict[reward_type] = reward_param
            args_dict["-df"] = df
            args_dict["-nw"] = 32
            args_dict["-np"] = 2
            args_dict["--continuous"] = 10000
            args_dict["-sl"] = 128
            args_dict["-clr"] = 1e-4
            args_dict["-mlr"] = 1e-4
            args_dict["-alr"] = 1e-4
            args_dict["-rl"] = "a3c"
            args_dict["-e"] = "../environments/two_arms_45_max_torque_1000_ndiscrete_128_medium_weight_balls_dpi_10.pkl"
            args_dict["--model-buffer-size"] = 1
            args_dict["--entropy-reg"] = 0.005
            commands.append(make_command(path, args_dict))
            paths.append(path)
    return commands, paths


def array_discrete_a3c_search_entropy_reg():
    commands, paths = [], []
    for reg in [0.0, 0.0005, 0.001, 0.005]:
        # for reward_type, reward_param in [("--minimize-pred-err", [""]),
        #                                   ("--maximize-pred-err", [""]),
        #                                   ("--target-pred-err", [0.010])]:
        for reward_type, reward_param in [("--target-pred-err", [0.005]),
                                          ("--target-pred-err", [0.010]),
                                          ("--target-pred-err", [0.015])]:
            path = array_path + "/{}_{}_{}".format(
                format_reward_type(reward_type),
                format_reward_params(*reward_param),
                reg)
            args_dict = {}
            args_dict[reward_type] = reward_param
            args_dict["-df"] = 0.85
            args_dict["-nw"] = 32
            args_dict["-np"] = 2
            args_dict["--continuous"] = 10000
            args_dict["-se"] = args_dict["--continuous"]
            args_dict["-sl"] = 64
            args_dict["-clr"] = 1e-4
            args_dict["-mlr"] = 1e-4
            args_dict["-alr"] = 1e-4
            args_dict["-rl"] = "a3c"
            args_dict["-e"] = "../environments/two_arms_45_max_torque_1000_ndiscrete_128_medium_weight_balls_dpi_10.pkl"
            args_dict["--model-buffer-size"] = 1
            args_dict["--entropy-reg"] = reg
            commands.append(make_command(path, args_dict))
            paths.append(path)
    return commands, paths


def array_standard_experiments_good_params():
    commands, paths = [], []
    for reward_type, reward_param in [("--minimize-pred-err", [""]),
                                      ("--maximize-pred-err", [""]),
                                      ("--target-pred-err", [0.005]),
                                      ("--target-pred-err", [0.010]),
                                      ("--target-pred-err", [0.015])]:
        path = array_path + "/{}_{}".format(
            format_reward_type(reward_type),
            format_reward_params(*reward_param))
        args_dict = {}
        args_dict[reward_type] = reward_param
        args_dict["-df"] = 0.85
        args_dict["-nw"] = 32
        args_dict["-np"] = 2
        args_dict["--continuous"] = 50000
        args_dict["-se"] = args_dict["--continuous"]
        args_dict["-sl"] = 32
        args_dict["-clr"] = 5e-5
        args_dict["-mlr"] = 1e-4
        args_dict["-alr"] = 5e-5
        args_dict["-rl"] = "a3c"
        args_dict["-e"] = "../environments/two_arms_45_max_torque_1000_ndiscrete_128_medium_weight_balls_dpi_10.pkl"
        args_dict["--model-buffer-size"] = 1
        args_dict["--entropy-reg"] = 0.005
        commands.append(make_command(path, args_dict))
        paths.append(path)
    return commands, paths


def array_pee_discrete_a3c():
    commands, paths = [], []
    for df in [0, 0.5, 0.85]:
        for alr in [1e-5, 5e-5, 1e-4]:
            path = array_path + "/alr_clr_{}_mlr_1e-4_df_{}".format(
                format_learning_rate(alr),
                format_discount_factor(df))
            args_dict = {}
            args_dict["--pred-err-err"] = 0.003
            args_dict["-df"] = df
            args_dict["-nw"] = 32
            args_dict["-np"] = 2
            args_dict["--continuous"] = 10000
            args_dict["-sl"] = 128
            args_dict["-clr"] = alr
            args_dict["-mlr"] = 1e-4
            args_dict["-alr"] = alr
            args_dict["-rl"] = "a3c"
            args_dict["-e"] = "../environments/two_arms_45_max_torque_1000_ndiscrete_128_medium_weight_balls.pkl"
            args_dict["--model-buffer-size"] = 1
            args_dict["--entropy-reg"] = 0.005
            commands.append(make_command(path, args_dict))
            paths.append(path)
    return commands, paths


commands, paths = array_standard_experiments_good_params()
if array_id < len(commands):
    cmd = commands[array_id]
    experiment_path = paths[array_id]
    print("array ID = {}\t\t{}".format(array_id, cmd))
    os.system(cmd)
    # os.rename(slurm_log_path, experiment_path + "/out.txt")
