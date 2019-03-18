import sys
import asynchronous as ac
import environment
import argparse
import pickle
import json


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
    help="Stage-wise setting. Specify nNumber of stages to compute (one stage = model + rl)"
)

parser.add_argument(
    '--model-stage-length',
    type=int,
    action='store',
    default=1000,
    help="Stage-wise setting. Specify number of model iterations"
)

parser.add_argument(
    '--critic-stage-length',
    type=int,
    action='store',
    default=500,
    help="Stage-wise setting. Specify number of critic alone iterations"
)

parser.add_argument(
    '--actor-stage-length',
    type=int,
    action='store',
    default=2500,
    help="Stage-wise setting. Specify number of rl iterations"
)

parser.add_argument(
    '-c', '--continuous',
    type=int,
    action='store',
    default=None,
    help="Train continuously. Pass the number of iteration."
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

parser.add_argument(
    '--range-pred-err',
    type=float,
    nargs=2,
    action='store',
    help="Reward type. Pass the 'range_mini' and 'range_maxi' params",
    default=None
)

parser.add_argument(
    '--pred-err-err',
    type=float,
    action='store',
    help="Reward type. Pass the 'pee_model_loss_converges_to' params",
    default=None
)

parser.add_argument(
    '-mlr', '--model-lr',
    type=float,
    action='store',
    help="Model learning rate",
    default=1e-4
)

parser.add_argument(
    '-clr', '--critic-lr',
    type=float,
    action='store',
    help="Critic learning rate",
    default=1e-3
)

parser.add_argument(
    '-alr', '--actor-lr',
    type=float,
    action='store',
    help="Actor learning rate",
    default=1e-5
)

parser.add_argument(
    '-mbs', '--model-buffer-size',
    type=int,
    action='store',
    help="Model buffer size",
    default=1
)

parser.add_argument(
    '-rl', '--rl-algo',
    type=str,
    choices=["DPG", "A3C", "dpg", "a3c"],
    action='store',
    help="Type of rl algorithm.",
    default="DPG"
)


args = parser.parse_args()
with open(args.environment_conf, "rb") as f:
    args_env = pickle.load(f)
if args.minimize_pred_err is not None:
    RewardCls = ac.MinimizeJointAgentWorker
    reward_params = {"model_loss_converges_to": args.minimize_pred_err}
elif args.maximize_pred_err is not None:
    RewardCls = ac.MaximizeJointAgentWorker
    reward_params = {"model_loss_converges_to": args.maximize_pred_err}
elif args.target_pred_err is not None:
    RewardCls = ac.TargetErrorJointAgentWorker
    reward_params = {"target_prediction_error": args.target_pred_err}
elif args.range_pred_err is not None:
    RewardCls = ac.RangeErrorJointAgentWorker
    reward_params = {"range_mini": args.range_pred_err[0], "range_maxi": args.range_pred_err[1]}
elif args.pred_err_err is not None:
    RewardCls = ac.PEEJointAgentWorker
    reward_params = {"pee_model_loss_converges_to": args.pred_err_err}
else:
    RewardCls = ac.MinimizeJointAgentWorker
    reward_params = {"model_loss_converges_to": 0.043}

if args.rl_algo.lower() == "dpg":
    RLCls = ac.DPGJointAgentWorker
elif args.rl_algo.lower() == "a3c":
    raise NotImplementedError("Implement A3C please")
    RLCls = ac.A3CJointAgentWorker
else:
    raise ValueError("Check the  --rl-algo  argument please")


WorkerCls = type('WorkerCls', (RewardCls, RLCls), {})

args_worker = (args.discount_factor, args.sequence_length, reward_params, args.model_lr, args.critic_lr, args.actor_lr, args.sequence_length * args.model_buffer_size)

with ac.Experiment(
        args.n_parameter_servers, args.n_workers, WorkerCls,
        args.path, args_env, args_worker, display_dpi=3) as experiment:
    ### save command line
    with open(args.path + "/cmd.txt", "w") as f:
        f.write("python3 " + " ".join(sys.argv))
    with open(args.path + "/parameters.json", "w") as f:
        json.dump(args.__dict__, fp=f, indent=4, sort_keys=True)
    ### start display if needed
    if args.display:
        experiment.start_display_worker(training=False)
        experiment.start_tensorboard()
    ### Save initial weights
    experiment.save_model("initial")
    if args.continuous:
        ### continuous learning
        done = 0
        save_every = min(args.continuous, 20000)
        i = 0
        while done < args.continuous:
            experiment.asynchronously_run_both(save_every, "continuous")
            done += save_every
            experiment.save_model("{}".format(i))
            i += 1
    else:
        ### stage-wise learning
        for i in range(args.n_stages):
                experiment.asynchronously_run_model(args.model_stage_length)
                experiment.save_model("after_model_{}".format(i))
                experiment.asynchronously_run_reinforcement_learning(args.critic_stage_length, train_actor=False)
                experiment.asynchronously_run_reinforcement_learning(args.actor_stage_length, train_actor=True)
                experiment.save_model("after_rl_{}".format(i))
