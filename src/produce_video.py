if __name__ == "__main__":
    from tempfile import TemporaryDirectory
    from shutil import rmtree
    from asynchronous import Experiment
    import pickle
    import argparse
    import os

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'path', metavar="PATH",
        type=str,
        action='store',
        help="Path to the checkpoint."
    )
    parser.add_argument(
        '--name',
        type=str,
        default="final",
        help="Name for the videos."
    )
    parser.add_argument(
        '-tsf', "--time-scale-factor",
        default=None,
        type=int,
        help="Overwrite time scale factor"
    )

    args = parser.parse_args()
    checkpoint_path = args.path
    experiment_path = os.path.abspath(checkpoint_path + "/../../")

    with open(experiment_path + "/conf/conf.pkl", "rb") as f:
        conf = pickle.load(f)

    if args.time_scale_factor is not None:
        conf.worker_conf.time_scale_factor = args.time_scale_factor

    with TemporaryDirectory() as tmppath:
        with Experiment(1, 1, tmppath + "/replay", conf, display_dpi=3) as experiment:
            experiment.restore_model(checkpoint_path)
            experiment.fill_goal_buffer()
            experiment.save_video(args.name + ".mp4", path=experiment_path + "/video", length_in_sec=120)
