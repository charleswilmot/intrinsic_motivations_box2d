from conf import Conf
import time
from shutil import rmtree
from asynchronous import Experiment
import pickle


default_lr = 1e-4
default_discount_factor = 0.9


def bool_helper(thing):
    return thing in ["True", "true", 1, "1"]


def make_experiment_path(date=None, lr=None, discount_factor=None, description=None):
    date = date if date else time.strftime("%Y_%m_%d-%H.%M.%S", time.localtime())
    lr = lr if lr else default_lr
    discount_factor = discount_factor if discount_factor else default_discount_factor
    description = ("__" + description) if description else ""
    experiment_dir = "../experiments/{}_lr{:.2e}_discount_factor{:.2e}{}".format(
        date, lr, discount_factor, description)
    return experiment_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-ep', '--experiment-path',
        type=str,
        action='store',
        help="Path to the experiment directory."
    )
    parser.add_argument(
        '-np', '--n-parameter-servers',
        type=int,
        help="Number of parameter servers.",
        default=1
    )
    parser.add_argument(
        '-nw', '--n-workers',
        type=int,
        help="Number of workers.",
        default=4
    )
    parser.add_argument(
        '-d', '--description',
        type=str,
        help="Short description of the experiment.",
        default=""
    )
    parser.add_argument(
        '-t', '--tensorboard',
        action="store_true",
        help="Start TensorBoard."
    )
    parser.add_argument(
        '-dp', '--display',
        action="store_true",
        help="Display environment."
    )
    parser.add_argument(
        '-se', '--save-every',
        type=int,
        help="Save every N updates.",
        default=100000
    )
    parser.add_argument(
        '-nt', '--n-trajectories',
        type=int,
        help="Number of trajectories to be simulated.",
        default=500000
    )
    parser.add_argument(
        '-sl', '--sequence-length',
        type=int,
        help="Length of an episode.",
        default=100
    )
    parser.add_argument(
        '-u', '--updates-per-episode',
        type=int,
        help="Number of updates per gathered trajectory.",
        default=1
    )
    parser.add_argument(
        '-tsf', '--time-scale-factor',
        type=int,
        help="Number of child updates per father updates.",
        default=10
    )
    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        help="Number of trajectories in a batch.",
        default=1
    )
    parser.add_argument(
        '-r', '--restore-from',
        type=str,
        default="",
        help="Checkpoint to restore from."
    )
    parser.add_argument(
        '-lr', '--learning-rate',
        type=float,
        default=default_lr,
        help="learning rate."
    )
    parser.add_argument(
        '-asr', '--actor-speed-ratio',
        type=float,
        default=10,
        help="How much slower is the actor?"
    )
    parser.add_argument(
        '-tae', '--train-actor-every',
        type=int,
        default=10,
        help="How often is the actor trained?"
    )
    parser.add_argument(
        '-tse', '--train-state-every',
        type=int,
        default=10,
        help="How often is the state trained?"
    )
    parser.add_argument(
        '-df', '--discount-factor',
        type=float,
        default=default_discount_factor,
        help="Discount factor."
    )
    parser.add_argument(
        '-bs', '--buffer-size',
        type=int,
        default=1000,
        help="Number of transitions that can be stored in the replay buffer."
    )
    parser.add_argument(
        '-jm', '--json-model',
        type=str,
        default="../models/two_arms_max_torque_1000_medium_weight_balls.json",
        help="Path to the json model of the environment."
    )
    parser.add_argument(
        '-so', '--skin-order',
        type=str,
        default="../models/two_arms_skin_order.pkl",
        help="Path to a pickle file containing the skin order (list of body_name / edge index tuples)."
    )
    parser.add_argument(
        '-sr', '--skin-resolution',
        type=int,
        default=20,
        help="Rsolution of the skin (number of sensors)."
    )
    parser.add_argument(
        '-xlim', '--xlim',
        type=list,
        default=[-20.5, 20.5],
        help="X limits of the environment."
    )
    parser.add_argument(
        '-ylim', '--ylim',
        type=list,
        default=[-13.5, 13.5],
        help="Y limits of the environment."
    )
    parser.add_argument(
        '-dpi', '--dpi',
        type=int,
        default=10,
        help="DPI of the vision sensor."
    )
    parser.add_argument(
        '-esl', '--env-step-length',
        type=int,
        default=45,
        help="Number of step simulated n the environment per RL step."
    )
    parser.add_argument(
        '-dt', '--dt',
        type=float,
        default=1 / 150,
        help="Time resolution of the simulation."
    )
    parser.add_argument(
        '-nd', '--n-discrete',
        type=int,
        default=128,
        help="Parameter controling the discretization of the join position / speeds."
    )
    parser.add_argument(
        '-v', '--video',
        type=bool_helper,
        default=False,
        help="Should generate videos?"
    )
    parser.add_argument(
        '-tau', '--tau',
        type=float,
        default=0.05,
        help="Target update rate."
    )
    parser.add_argument(
        '-bns', '--behaviour-noise-scale',
        type=float,
        default=0.05,
        help="Time resolution of the simulation."
    )
    parser.add_argument(
        '-tsns', '--target-smoothing-noise-scale',
        type=float,
        default=0.005,
        help="Time resolution of the simulation."
    )
    parser.add_argument(
        '-gbf', '--goal-buffer-size',
        type=int,
        default=500,
        help="Size of the buffer from which goals are sampled."
    )
    parser.add_argument(
        '-a', '--agency-conf-path',
        type=str,
        default="../agencies/simple_agency.txt",
        help="Path to the agency conf file."
    )

    args = parser.parse_args()

    if not args.experiment_path:
        args.experiment_path = make_experiment_path(
            lr=args.learning_rate,
            discount_factor=args.discount_factor,
            description=args.description)

    conf = Conf.from_args(args)

    with Experiment(args.n_parameter_servers, args.n_workers, args.experiment_path, conf, display_dpi=3) as experiment:
        if args.restore_from:
            experiment.restore_model(args.restore_from)
        if args.display:
            experiment.start_display_worker(training=True)
        if args.tensorboard:
            experiment.start_tensorboard()
        experiment.randomize_env(n=100)
        for i in range(args.n_trajectories // args.save_every):
            experiment.asynchronously_train(args.save_every)
            experiment.save_model()
        if args.video:
            experiment.save_video("final.mp4")
