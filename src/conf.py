import json


class ConfEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ConfRoot):
            dictionary = {}
            for key, val in obj.__dict__.items():
                if isinstance(key, ConfRoot):
                    dictionary[key] = self.default(val)
                else:
                    dictionary[key] = val
            return dictionary
        return json.JSONEncoder.default(self, obj)


class ConfRoot:
    def dump(self, file_object):
        json.dump(self, file_object, cls=ConfEncoder, indent=4, sort_keys=True)


class WorkerConf(ConfRoot):
    def __init__(self, discount_factor, sequence_length, critic_learning_rate, epsilon_init, epsilon_decay,
                       buffer_size, updates_per_episode, batch_size, her_strategy, n_actions, parametrization_type):
        self.discount_factor = discount_factor
        self.sequence_length = sequence_length
        self.critic_learning_rate = critic_learning_rate
        self.epsilon_init = epsilon_init
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.updates_per_episode = updates_per_episode
        self.batch_size = batch_size
        self.her_strategy = her_strategy
        self.n_actions = n_actions
        self.parametrization_type = parametrization_type

    @staticmethod
    def from_args(args):
        return WorkerConf(args.discount_factor,
                          args.sequence_length,
                          args.critic_learning_rate,
                          args.epsilon_init,
                          args.epsilon_decay,
                          args.buffer_size,
                          args.updates_per_episode,
                          args.batch_size,
                          args.her_strategy,
                          args.n_actions,
                          args.parametrization_type)


class GoalLibraryConf(ConfRoot):
    def __init__(self, softmax_temperature, min_reaching_prob, goal_library_size, ema_speed, goal_sampling_type):
        self.softmax_temperature = softmax_temperature
        self.min_reaching_prob = min_reaching_prob
        self.goal_library_size = goal_library_size
        self.ema_speed = ema_speed
        self.goal_sampling_type = goal_sampling_type

    @staticmethod
    def from_args(args):
        return GoalLibraryConf(args.softmax_temperature,
                               args.min_reaching_prob,
                               args.goal_library_size,
                               args.ema_speed,
                               args.goal_sampling_type)


class EnvironmentConf(ConfRoot):
    def __init__(self, json_model, skin_order, skin_resolution, xlim, ylim, dpi, env_step_length, dt, n_discrete):
        self.json_model = json_model
        self.skin_order = skin_order
        self.skin_resolution = skin_resolution
        self.xlim = xlim
        self.ylim = ylim
        self.dpi = dpi
        self.env_step_length = env_step_length
        self.dt = dt
        self.n_discrete = n_discrete

    @staticmethod
    def from_args(args):
        return EnvironmentConf(args.json_model,
                               args.skin_order,
                               args.skin_resolution,
                               args.xlim,
                               args.ylim,
                               args.dpi,
                               args.env_step_length,
                               args.dt,
                               args.n_discrete)


class Conf(ConfRoot):
    def __init__(self, worker_conf, goal_library_conf, environment_conf):
        self.worker_conf = worker_conf
        self.goal_library_conf = goal_library_conf
        self.environment_conf = environment_conf

    @staticmethod
    def from_args(args):
        worker_conf = WorkerConf.from_args(args)
        goal_library_conf = GoalLibraryConf.from_args(args)
        environment_conf = EnvironmentConf.from_args(args)
        return Conf(worker_conf, goal_library_conf, environment_conf)


if __name__ == "__main__":
    a = WorkerConf(1,2,3,4,5,6,7,8,9)
    b = GoalLibraryConf(1,2,3,4)
    c = EnvironmentConf(1,2,3,4,5,6,7,8,9)
    d = Conf(a, b, c)
    with open("/tmp/deleteme.txt", "w") as f:
        d.dump(f)

    import os
    os.system("cat /tmp/deleteme.txt")
