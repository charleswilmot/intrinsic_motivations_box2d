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
        '-n', "--n-workers",
        type=int,
        default=10,
        help="Number of workers."
    )
    parser.add_argument(
        '--name',
        type=str,
        default="final",
        help="Name for the videos."
    )

    args = parser.parse_args()
    checkpoint_path = args.path
    experiment_path = os.path.abspath(checkpoint_path + "/../../")

    with open(experiment_path + "/conf/conf.pkl", "rb") as f:
        conf = pickle.load(f)

    with TemporaryDirectory() as tmppath:
        with Experiment(1, args.n_workers, tmppath + "/replay", conf, display_dpi=3) as experiment:
            experiment.restore_model(checkpoint_path)
            experiment.restore_goal_library(experiment_path + "/goals/dumps/worker_0.pkl")
            # experiment.save_video(args.name, path=experiment_path + "/video", n_frames=2000)
            experiment.save_video_all_goals(args.name, path=experiment_path + "/video", n_frames=150)
