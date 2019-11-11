from conf import Conf
import time
from shutil import rmtree
from asynchronous import Experiment
import pickle


default_clr = 1e-4
default_epsilon = 0.05
default_discount_factor = 0.9


def bool_helper(thing):
    return thing in ["True", "true", 1, "1"]


def make_experiment_path(date=None, clr=None, epsilon=None, discount_factor=None, description=None):
    date = date if date else time.strftime("%Y_%m_%d-%H.%M.%S", time.localtime())
    clr = clr if clr else default_clr
    epsilon = epsilon if epsilon else default_epsilon
    discount_factor = discount_factor if discount_factor else default_discount_factor
    description = ("__" + description) if description else ""
    experiment_dir = "../experiments/{}_clr{:.2e}_epsilon{:.2e}_discount_factor{:.2e}{}".format(
        date, clr, epsilon, discount_factor, description)
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
        '-fe', '--flush-every',
        type=int,
        help="Flush every N simulated trajectory.",
        default=20000
    )
    parser.add_argument(
        '-nt', '--n-trajectories',
        type=int,
        help="Number of trajectories to be simulated.",
        default=10000000
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
        '-clr', '--critic-learning-rate',
        type=float,
        default=default_clr,
        help="critic learning rate."
    )
    parser.add_argument(
        '-df', '--discount-factor',
        type=float,
        default=default_discount_factor,
        help="Discount factor."
    )
    parser.add_argument(
        '-st', '--softmax-temperature',
        type=float,
        default=1.0,
        help="Temperature of the softmax sampling."
    )
    parser.add_argument(
        '-her', '--her-strategy',
        type=str,
        default="all_contacts",
        help="Strategy for the HER."
    )
    parser.add_argument(
        '-pt', '--parametrization-type',
        type=str,
        default="none",
        help="Parametrization type. ('none', 'concat' or 'affine', case independent)"
    )
    parser.add_argument(
        '-bs', '--buffer-size',
        type=int,
        default=100,
        help="Number of trajectories that can be stored in the replay buffer."
    )
    parser.add_argument(
        '-ema', '--ema-speed',
        type=float,
        default=0.99999,
        help="Exponential moving average speed for the goal library (goal library)."
    )
    parser.add_argument(
        '-mrp', '--min-reaching-prob',
        type=float,
        default=0.0005,
        help="Minimum probability of reaching when the goal is not in pursued for a goal to be sampled."
    )
    parser.add_argument(
        '-gls', '--goal-library-size',
        type=int,
        default=100,
        help="Number of goals in the library."
    )
    parser.add_argument(
        '-gst', '--goal-sampling-type',
        type=str,
        default="uniform",
        help="Type of sampling in the goal library."
    )
    parser.add_argument(
        '-ei', '--epsilon-init',
        type=float,
        default=default_epsilon,
        help="Initialization for the sampling rate."
    )
    parser.add_argument(
        '-ed', '--epsilon-decay',
        type=float,
        default=1.0,
        help="Decay for the sampling rate."
    )
    parser.add_argument(
        '-na', '--n-actions',
        type=int,
        default=11,
        help="Number of actions in the action set."
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
        default=True,
        help="Should generate videos?"
    )
    args = parser.parse_args()

    if not args.experiment_path:
        args.experiment_path = make_experiment_path(
            clr=args.critic_learning_rate,
            alr=args.actor_learning_rate,
            er=args.entropy_reg,
            description=args.description)

    conf = Conf.from_args(args)

    with Experiment(args.n_parameter_servers, args.n_workers, args.experiment_path, conf, display_dpi=3) as experiment:
        if args.restore_from:
            experiment.restore_model(args.restore_from)
            experiment.restore_goal_library(args.restore_from + "/../../goals/dumps/worker_0.pkl")
        if args.display:
            experiment.start_display_worker(training=True)
        if args.tensorboard:
            experiment.start_tensorboard()
        experiment.randomize_env(n=100)
        for i in range(args.n_trajectories // args.flush_every):
            experiment.asynchronously_train(args.flush_every)
            experiment.take_goal_library_snapshot()
            if (i + 1) % 10 == 0:
                experiment.print_goal_library()
                experiment.dump_goal_library()
            if (i + 1) % 100 == 0:
                experiment.save_model()
        experiment.take_goal_library_snapshot()
        experiment.print_goal_library()
        experiment.dump_goal_library()
        if args.video:
            experiment.save_video("final")
            experiment.save_video_all_goals("bragging", n_frames=150)
        if not (i + 1) % 100 == 0:
            experiment.save_model()
