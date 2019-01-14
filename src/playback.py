import argparse
from asynchronous import Experiment, get_cluster, JointAgentWorker
import environment
import time
from tempfile import TemporaryDirectory


parser = argparse.ArgumentParser()
parser.add_argument(
    'path', metavar="PATH",
    type=str,
    action='store',
    help="Path to the model to restore."
)

parser.add_argument(
    '-df', '--discount-factor',
    type=float,
    action='store',
    default=0.0
)

args = parser.parse_args()

skin_order = [
    ("Arm1_Left", 0),
    ("Arm2_Left", 0),
    ("Arm2_Left", 1),
    ("Arm2_Left", 2),
    ("Arm1_Left", 2),
    ("Arm1_Right", 0),
    ("Arm2_Right", 0),
    ("Arm2_Right", 1),
    ("Arm2_Right", 2),
    ("Arm1_Right", 2)]
skin_resolution = 12
xlim = [-20.5, 20.5]
ylim = [-13.5, 13.5]
json_model = "../models/two_arms.json"
dpi = 1
dt = 1 / 150
n_discrete = 32

logdir = TemporaryDirectory()

discount_factor = args.discount_factor
env_step_length = 45
sequence_length = 128  # 1024  # 64

N_WORKERS = 1
N_PARAMETER_SERVERS = 1

cluster = get_cluster(N_PARAMETER_SERVERS, N_WORKERS)
args_env = (json_model, skin_order, skin_resolution, xlim, ylim, dpi, dt, n_discrete)
args_worker = (discount_factor, env_step_length, sequence_length)

with Experiment(
        N_PARAMETER_SERVERS, N_WORKERS, JointAgentWorker,
        logdir.name + "/dummy/", args_env, args_worker, display_dpi=3) as experiment:
    experiment.restore_model(args.path)
    experiment.start_display_worker()
    time.sleep(60 * 5)
