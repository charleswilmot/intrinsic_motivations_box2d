import sys
import asynchronous as ac
import environment
import argparse
import pickle


parser = argparse.ArgumentParser()
parser.add_argument(
    'path', metavar="PATH",
    type=str,
    action='store',
    help="Path to the model to save directory. (Must not exist)"
)

parser.add_argument(
    '-df', '--discount-factor',
    type=float,
    action='store',
    default=0.0,
    help="Discount factor for the experiment."
)

parser.add_argument(
    '--display',
    action='store_true',
    help="If this flag is set, one worker will be used for displaying."
)

parser.add_argument(
    '-s', '--n-stages',
    type=int,
    action='store',
    default=4,
    help="Number of stages to compute (one stage = model + rl)"
)

parser.add_argument(
    '-np', '--n-parameter-servers',
    type=int,
    action='store',
    default=4,
    help="number of parameter servers"
)

parser.add_argument(
    '-nw', '--n-workers',
    type=int,
    action='store',
    default=16,
    help="number of workers"
)

parser.add_argument(
    '-sl', '--sequence-length',
    type=int,
    action='store',
    default=128,
    help="Number of rl steps to be gathered before updating the server."
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


args = parser.parse_args()
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
else:
    WorkerCls = ac.MinimizeJointAgentWorker
    reward_params = {"model_loss_converges_to": 0.043}

args_worker = (args.discount_factor, args.sequence_length, reward_params)

with ac.Experiment(
        args.n_parameter_servers, args.n_workers, WorkerCls,
        args.path, args_env, args_worker, display_dpi=3) as experiment:
    with open(args.path + "/cmd.txt", "w") as f:
        f.write("python3 " + " ".join(sys.argv))
    if args.display:
        experiment.start_display_worker()
        experiment.start_tensorboard()
    experiment.save_model("initial")
    for i in range(args.n_stages):
            summary_prefix = "stage_{}".format(i)
            experiment.asynchronously_run_model(4000 if i == 0 else 1000, summary_prefix)
            # experiment.asynchronously_run_model(500, summary_prefix)
            experiment.save_model("after_model_{}".format(i))
            experiment.asynchronously_run_reinforcement_learning(500, summary_prefix, train_actor=False)
            experiment.asynchronously_run_reinforcement_learning(2500, summary_prefix, train_actor=True)
            experiment.save_model("after_rl_{}".format(i))
