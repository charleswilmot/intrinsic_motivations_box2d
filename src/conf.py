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
    def __init__(self, discount_factor, learning_rate, actor_speed_ratio, train_actor_every, buffer_size,
                       updates_per_episode, time_scale_factor, batch_size, tau, behaviour_noise_scale,
                       target_smoothing_noise_scale, goal_buffer_size, agency_conf_path):
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.actor_speed_ratio = actor_speed_ratio
        self.train_actor_every = train_actor_every
        self.buffer_size = buffer_size
        self.updates_per_episode = updates_per_episode
        self.time_scale_factor = time_scale_factor
        self.batch_size = batch_size
        self.tau = tau
        self.behaviour_noise_scale = behaviour_noise_scale
        self.target_smoothing_noise_scale = target_smoothing_noise_scale
        self.goal_buffer_size = goal_buffer_size
        self.agency_conf_path = agency_conf_path

    @staticmethod
    def from_args(args):
        return WorkerConf(args.discount_factor,
                          args.learning_rate,
                          args.actor_speed_ratio,
                          args.train_actor_every,
                          args.buffer_size,
                          args.updates_per_episode,
                          args.time_scale_factor,
                          args.batch_size,
                          args.tau,
                          args.behaviour_noise_scale,
                          args.target_smoothing_noise_scale,
                          args.goal_buffer_size,
                          args.agency_conf_path)


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
    def __init__(self, worker_conf, environment_conf):
        self.worker_conf = worker_conf
        self.environment_conf = environment_conf

    @staticmethod
    def from_args(args):
        worker_conf = WorkerConf.from_args(args)
        environment_conf = EnvironmentConf.from_args(args)
        return Conf(worker_conf, environment_conf)


if __name__ == "__main__":
    a = WorkerConf(1,2,3,4,5,6,7,8,9)
    b = GoalLibraryConf(1,2,3,4)
    c = EnvironmentConf(1,2,3,4,5,6,7,8,9)
    d = Conf(a, b, c)
    with open("/tmp/deleteme.txt", "w") as f:
        d.dump(f)

    import os
    os.system("cat /tmp/deleteme.txt")
