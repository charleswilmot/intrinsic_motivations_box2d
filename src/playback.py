import pickle
import re
import os
import argparse
import asynchronous as ac
import environment
import time
from tempfile import TemporaryDirectory


parser = argparse.ArgumentParser()
parser.add_argument(
    'path', metavar="PATH",
    type=str,
    action='store',
    help="Path to the model to restore."
)

parser.add_argument(
    '-df', '--discount-factor',
    type=float,
    action='store',
    default=0.0
)

parser.add_argument(
    '-rl', '--after-rl-only',
    action='store_true'
)

parser.add_argument(
    '-s', '--sample',
    action='store_true'
)

parser.add_argument(
    '-e', '--environment-conf',
    type=str,
    action='store',
    help="Path to the environment config file.",
    default="../environments/two_arms_45.pkl"
)

parser.add_argument(
    '--minimize-pred-err',
    type=float,
    action='store',
    help="Reward type. Pass the 'model_loss_converges_to' param",
    default=None
)

parser.add_argument(
    '--maximize-pred-err',
    type=float,
    action='store',
    help="Reward type. Pass the 'model_loss_converges_to' param",
    default=None
)

parser.add_argument(
    '--target-pred-err',
    type=float,
    action='store',
    help="Reward type. Pass the 'target_prediction_error' param",
    default=None
)

parser.add_argument(
    '--range-pred-err',
    type=float,
    nargs=2,
    action='store',
    help="Reward type. Pass the 'range_mini' and 'range_maxi' params",
    default=None
)


SEQUENCE_LENGTH = 8
N_WORKERS = 2
N_PARAMETER_SERVERS = 1
logdir = TemporaryDirectory()

args = parser.parse_args()
# get the Worker class to be used and the reward parameters
with open(args.environment_conf, "rb") as f:
    args_env = pickle.load(f)
if args.minimize_pred_err is not None:
    WorkerCls = ac.MinimizeJointAgentWorker
    reward_params = {"model_loss_converges_to": args.minimize_pred_err}
elif args.maximize_pred_err is not None:
    WorkerCls = ac.MaximizeJointAgentWorker
    reward_params = {"model_loss_converges_to": args.maximize_pred_err}
elif args.target_pred_err is not None:
    WorkerCls = ac.TargetErrorJointAgentWorker
    reward_params = {"target_prediction_error": args.target_pred_err}
elif args.range_pred_err is not None:
    WorkerCls = ac.RangeErrorJointAgentWorker
    reward_params = {"range_mini": args.range_pred_err[0], "range_maxi": args.range_pred_err[1]}
else:
    WorkerCls = ac.MinimizeJointAgentWorker
    reward_params = {"model_loss_converges_to": 0.043}
# get the environment parameters
with open(args.environment_conf, "rb") as f:
    args_env = pickle.load(f)
args_worker = (args.discount_factor, SEQUENCE_LENGTH, reward_params)

models = [x for x in os.listdir(args.path) if os.path.isdir(args.path + "/" + x)]
initial = [x for x in models if re.match("initial", x)]
after_model = [x for x in models if re.match("after_model_[0-9]+", x)]
after_rl = [x for x in models if re.match("after_rl_[0-9]+", x)]
continuous = [x for x in models if re.match("[0-9]+", x)]
after_model.sort(key=lambda x: int(re.match("after_model_([0-9]+)", x).group(1)))
after_rl.sort(key=lambda x: int(re.match("after_rl_([0-9]+)", x).group(1)))
continuous.sort(key=lambda x: int(re.match("([0-9]+)", x).group(1)))
ordered_models = initial + [val for pair in zip(after_model, after_rl) for val in pair]

if args.after_rl_only:
    models = initial + after_rl + continuous
else:
    models = ordered_models + continuous

with ac.Experiment(
        N_PARAMETER_SERVERS, N_WORKERS, WorkerCls,
        logdir.name + "/dummy/", args_env, args_worker, display_dpi=3) as experiment:
    experiment.start_display_worker(sample=args.sample)
    for model in models:
        print(model)
        experiment.restore_model(args.path + "/" + model)
        while input(">>> ") != "n":
            pass
