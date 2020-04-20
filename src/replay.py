if __name__ == "__main__":
    from tempfile import TemporaryDirectory
    import time
    from shutil import rmtree
    from asynchronous import Experiment
    import pickle
    import argparse
    from training import bool_helper

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'path', metavar="PATH",
        type=str,
        action='store',
        help="Path to the checkpoint."
    )

    parser.add_argument(
        '-t', "--training",
        nargs='+',
        type=bool_helper,
        help="Training per level"
    )

    parser.add_argument(
        '-tsf', "--time-scale-factor",
        default=None,
        type=int,
        help="Overwrite time scale factor"
    )


    args = parser.parse_args()
    checkpoint_path = args.path
    experiment_path = checkpoint_path + "/../../"

    with open(experiment_path + "/conf/conf.pkl", "rb") as f:
        conf = pickle.load(f)

    if args.time_scale_factor is not None:
        conf.worker_conf.time_scale_factor = args.time_scale_factor

    with TemporaryDirectory() as tmppath:
        with Experiment(1, 1, tmppath + "/replay", conf, display_dpi=3) as experiment:
            experiment.restore_model(checkpoint_path)
            experiment.fill_goal_buffer()
            experiment.start_display_worker(training=args.training)
