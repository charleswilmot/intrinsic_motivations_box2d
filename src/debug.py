import time
from shutil import rmtree
from asynchronous import Experiment
import pickle

with open("../environments/two_arms_45_max_torque_1000_ndiscrete_128_medium_weight_balls_dpi_10_low_res.pkl", "rb") as f:
    args_env = pickle.load(f)



args_worker = [
    0.5,            # discount_factor
    20,              # sequence_length
    1e-4,            # critic_lr
    1e-5,            # actor_lr
    0.01,          # entropy_coef
    500,             # replay_buffer_size
    10,              # updates_per_episode
    "all_contacts"   # HER_strategy
]

rmtree("../experiments/skin/debug/", ignore_errors=True)

with Experiment(1, 8, "../experiments/skin/debug/", args_env, args_worker, display_dpi=3) as experiment:
    experiment.start_display_worker(training=True)
    for i in range(10000):
        experiment.asynchronously_train(1000, train_actor=(i > 20))
        experiment.take_goal_library_snapshot()
        if (i + 1) % 10 == 0:
            experiment.print_goal_library()
            experiment.dump_goal_library()
        if (i + 1) % 100 == 0:
            experiment.save_model()
        # experiment.randomize_env()
    # experiment.set_all_display_workers_idle()
    # experiment.save_vision_related_to_goals()
    # time.sleep(500)
    #experiment.save_video("{}_sample".format(done), 2 * 60 * 24 // args.sequence_length, True)
    #experiment.save_contact_logs(done)
