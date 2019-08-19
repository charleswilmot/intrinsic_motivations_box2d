if __name__ == "__main__":
    from tempfile import TemporaryDirectory
    import time
    from shutil import rmtree
    from asynchronous import Experiment
    import pickle
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'path', metavar="PATH",
        type=str,
        action='store',
        help="Path to the checkpoint."
    )

    args = parser.parse_args()
    checkpoint_path = args.path
    experiment_path = checkpoint_path + "/../../"

    with open(experiment_path + "conf/env_conf.pkl", "rb") as f:
        args_env = pickle.load(f)

    with open(experiment_path + "conf/worker_conf.pkl", "rb") as f:
        args_worker = pickle.load(f)

    with TemporaryDirectory() as tmppath:
        with Experiment(1, 1, tmppath + "/replay", args_env, args_worker, display_dpi=3) as experiment:
            experiment.restore_model(checkpoint_path)
            experiment.restore_goals_warehouse(experiment_path + "goals/dumps/")
            experiment.start_display_worker(training=True)
            time.sleep(600)
