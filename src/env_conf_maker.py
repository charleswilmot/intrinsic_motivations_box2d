import pickle


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
env_step_length = 450


args_env = (json_model, skin_order, skin_resolution, xlim, ylim, dpi, env_step_length, dt, n_discrete)

with open("../environments/two_arms_450.pkl", "wb") as f:
    pickle.dump(args_env, f)
