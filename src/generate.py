import database
import environment
import numpy as np
import argparse
import datetime
import time

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
parser = argparse.ArgumentParser()
parser.add_argument(
    '-dpi', '--dpi',
    type=int,
    action='store', default=1,
    help="Resolution of the vision sensor."
)

parser.add_argument(
    '-s', '--chunk-size',
    type=int,
    action='store', default=512 * 1e6,
    help="Maximum size of a chunk in bytes."
)

parser.add_argument(
    '-o', '--output',
    type=str, action='store', default='/tmp/box2d_{}'.format(st),
    help="Output directory. Must not exist."
)

parser.add_argument(
    '-t', '--action-type',
    type=str, choices=["speeds", "positions"], default="speeds",
    help="Control mode."
)

parser.add_argument(
    '-sf', '--simulation-freq',
    type=float,
    action='store', default=150,
    help="Frequence of the simulator."
)

parser.add_argument(
    '-re', '--record-every',
    type=int,
    action='store', default=15,
    help="Record every N simulation steps."
)

parser.add_argument(
    '-ae', '--action-every',
    type=int,
    action='store', default=5,
    help="Action every N recording."
)

parser.add_argument(
    '-n', '--n-record',
    type=int,
    action='store', default=100000,
    help="Number of records."
)


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
args = parser.parse_args()
env = environment.Environment(
    "../models/two_arms.json",
    skin_order,
    skin_resolution,
    xlim,
    ylim,
    dpi=args.dpi,
    dt=1 / args.simulation_freq)


filename_pattern = "sf{}_re{}_ae{}_".format(args.simulation_freq, args.record_every, args.action_every) + "chunk{}.tfr"

n_hash = -1
with database.DatabaseWriter(args.output, filename_pattern=filename_pattern, chunk_size=args.chunk_size) as write:
    for record_number in range(args.n_record):
        n_hash_new = int(20 * record_number / args.n_record)
        if n_hash_new != n_hash:
            print(" |" + "#" * n_hash_new + " " * (20 - n_hash_new) + "|", end='\r')
            n_hash = n_hash_new
        if record_number % args.action_every == 0:
            actions = {
                "Arm1_to_Arm2_Left": np.random.uniform(-1, 1),
                "Ground_to_Arm1_Left": np.random.uniform(-1, 1),
                "Arm1_to_Arm2_Right": np.random.uniform(-1, 1),
                "Ground_to_Arm1_Right": np.random.uniform(-1, 1)
            }
            env.set_positions(actions)
        write(*env.state, actions)
        for j in range(args.record_every):
            env.step()
