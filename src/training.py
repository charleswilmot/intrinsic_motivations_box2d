import time
from shutil import rmtree
from asynchronous import Experiment
import pickle


def make_experiment_path(date=None, clr=None, alr=None, er=None, description=None):
    date = date if date else time.strftime("%Y_%m_%d-%H.%M.%S", time.localtime())
    clr = clr if clr else default_clr
    alr = alr if alr else default_alr
    er = er if er else default_er
    description = ("__" + description) if description else ""
    experiment_dir = "../experiments/{}_clr{:.2e}_alr{:.2e}_entropy{:.2e}{}".format(
        date, clr, alr, er, description)
    return experiment_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    default_clr = 1e-4
    default_alr = 1e-5
    default_discount_factor = 0.5
    default_er = 0.01

    parser.add_argument(
        '-ep', '--experiment-path',
        type=str,
        action='store',
        default="",
        help="Path to the experiment directory."
    )
    parser.add_argument(
        '-np', '--n-parameter-servers',
        type=int,
        help="Number of parameter servers.",
        default=2
    )
    parser.add_argument(
        '-nw', '--n-workers',
        type=int,
        help="Number of workers.",
        default=8
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
        default=1000
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
        default=20
    )
    parser.add_argument(
        '-u', '--updates-per-episode',
        type=int,
        help="Number of updates per gathered trajectory.",
        default=10
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
        '-alr', '--actor-learning-rate',
        type=float,
        default=default_alr,
        help="actor learning rate."
    )
    parser.add_argument(
        '-df', '--discount-factor',
        type=float,
        default=default_discount_factor,
        help="Discount factor."
    )
    parser.add_argument(
        '-er', '--entropy-reg',
        type=float,
        default=default_er,
        help="Entropy regularizer."
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
        '-env', '--environment-path',
        type=str,
        default="../environments/two_arms_45_max_torque_1000_ndiscrete_128_medium_weight_balls_dpi_10_low_res.pkl",
        help="Path to an environment pickle file."
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
        default=0.999,
        help="Exponential moving average speed for the goal library (goal library)."
    )

    args = parser.parse_args()

    if not args.experiment_path:
        args.experiment_path = make_experiment_path(
            clr=args.critic_learning_rate,
            alr=args.actor_learning_rate,
            er=args.entropy_reg,
            description=args.description)


    with open(args.environment_path, "rb") as f:
        args_env = pickle.load(f)



    args_worker = [
        args.discount_factor,             # discount_factor
        args.sequence_length,             # sequence_length
        args.critic_learning_rate,        # critic_lr
        args.actor_learning_rate,         # actor_lr
        args.entropy_reg,                 # entropy_coef
        args.softmax_temperature,         # softmax_temperature
        args.buffer_size,                 # replay_buffer_size
        args.updates_per_episode,         # updates_per_episode
        args.her_strategy                 # HER_strategy
    ]

    with Experiment(
            args.n_parameter_servers,
            args.n_workers,
            args.experiment_path,
            args_env,
            args_worker,
            display_dpi=3) as experiment:
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
        experiment.save_model()
        
