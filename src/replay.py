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

    with open(experiment_path + "/conf/conf.pkl", "rb") as f:
        conf = pickle.load(f)

    with TemporaryDirectory() as tmppath:
        with Experiment(1, 1, tmppath + "/replay", conf, display_dpi=3) as experiment:
            experiment.restore_model(checkpoint_path)
            experiment.restore_goal_library(experiment_path + "goals/dumps/worker_0.pkl")
            experiment.start_display_worker(training=False)
            time.sleep(60000)
