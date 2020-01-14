import tensorflow as tf
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import layers
import numpy as np
import json


def d(a, b):
    return tf.reduce_sum((a - b) ** 2, axis=-1)


class FixKerasIssue(dict):
    pass


class AgencyConfDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        ret = {}
        for model_type in dct:
            if model_type == "policy_model":
                code = dct["policy_model"]
                ret["policy_model"] = self._to_model_class(code)
                ret["target_policy_model"] = self._to_model_class(code)
            elif model_type == "critic_model":
                code = dct["critic_model"]
                ret["critic_0_model"] = self._to_model_class(code)
                ret["target_critic_0_model"] = self._to_model_class(code)
                ret["critic_1_model"] = self._to_model_class(code)
                ret["target_critic_1_model"] = self._to_model_class(code)
            elif model_type in ["state_model", "readout_goal_model", "readout_state_model", "readout_gstate_model"]:
                ret[model_type] = self._to_model_class(dct[model_type])
        dct.update(ret)
        return dct

    def _to_model_class(self, cls_str):
        return eval(cls_str)


class MultiInputModel(layers.Layer):
    def __call__(self, *args, **kwargs):
        return super().__call__(args, **kwargs)

    def call(self, inputs, **kwargs):
        ret = self.my_call(*inputs, **kwargs)
        self._set_trainable_weights()
        self._set_batchnorm_op()
        return ret

    def _set_batchnorm_op(self):
        pass


class DebugPolicyModel(MultiInputModel):
    def __init__(self, name="policy"):
        super().__init__(name=name)
        self.concat = layers.Concatenate()
        self.dense = layers.Dense(10)
        self.batchnorm = tf.layers.BatchNormalization(scale=False, center=False)

    def my_call(self, state, goal, training=True):
        ret = self.concat([state, goal])
        ret = self.dense(ret)
        ret = self.batchnorm(ret, training=training)
        return ret

    def _set_trainable_weights(self):
        self._trainable_weights = self.dense.trainable_variables


class DebugStateModel(MultiInputModel):
    def __init__(self, name="state"):
        super().__init__(name=name)
        self.dense = layers.Dense(10)

    def my_call(self, state):
        return self.dense(state)

    def _set_trainable_weights(self):
        self._trainable_weights = self.dense.trainable_variables


class DebugCriticModel(MultiInputModel):
    def __init__(self, name="critic"):
        super().__init__(name=name)
        self.concat = layers.Concatenate()
        self.dense = layers.Dense(1)

    def my_call(self, state, goal, action):
        ret = self.concat([state, goal, action])
        ret = self.dense(ret)
        return ret

    def _set_trainable_weights(self):
        self._trainable_weights = self.dense.trainable_variables


class SimplePolicyModel(MultiInputModel):
    def __init__(self, l0_size, l1_size, l2_size, input_batchnorm=False, output_batchnorm=True, name="policy"):
        super().__init__(name=name)
        self.concat0 = layers.Concatenate()
        self.concat1 = layers.Concatenate()
        self.dense0 = layers.Dense(l0_size, activation=tf.nn.relu)
        self.dense1 = layers.Dense(l1_size, activation=tf.nn.relu)
        self.dense2 = layers.Dense(l2_size, activation=tf.tanh)
        self._input_batchnorm = input_batchnorm
        self._output_batchnorm = output_batchnorm
        if input_batchnorm:
            self.input_batchnorm = tf.layers.BatchNormalization(scale=False, center=False)
        if output_batchnorm:
            self.output_batchnorm = tf.layers.BatchNormalization(scale=False, center=False)

    def my_call(self, state, goal, training=True):
        ret = self.concat0([state, goal])
        if self._input_batchnorm:
            ret = self.input_batchnorm(ret, training=training)
        ret = self.dense0(ret)
        ret = self.concat1([state, goal, ret])
        ret = self.dense1(ret)
        ret = self.dense2(ret)
        if self._output_batchnorm:
            ret = self.output_batchnorm(ret, training=training)
        return ret

    def _set_trainable_weights(self):
        self._trainable_weights = []
        self._trainable_weights += self.dense0.trainable_variables
        self._trainable_weights += self.dense1.trainable_variables
        self._trainable_weights += self.dense2.trainable_variables

    def _set_batchnorm_op(self):
        self.batchnorm_op = []
        self.batchnorm_vars = []
        if self._input_batchnorm:
            self.batchnorm_op.append(self.input_batchnorm.updates)
            self.batchnorm_vars += self.input_batchnorm.variables
        if self._output_batchnorm:
            self.batchnorm_op.append(self.output_batchnorm.updates)
            self.batchnorm_vars += self.output_batchnorm.variables


class SimpleStateModel(MultiInputModel):
    def __init__(self, l0_size, l1_size, l2_size, input_batchnorm=False, output_batchnorm=True, name="state"):
        super().__init__(name=name)
        self.dense0 = layers.Dense(l0_size, activation=tf.nn.relu)
        self.dense1 = layers.Dense(l1_size, activation=tf.nn.relu)
        self.dense2 = layers.Dense(l2_size, activation=tf.nn.relu)
        self._input_batchnorm = input_batchnorm
        self._output_batchnorm = output_batchnorm
        if input_batchnorm:
            self.input_batchnorm = tf.layers.BatchNormalization(scale=False, center=False)
        if output_batchnorm:
            self.output_batchnorm = tf.layers.BatchNormalization(scale=False, center=False)

    def my_call(self, state, training=True):
        if self._input_batchnorm:
            ret = self.input_batchnorm(state, training=training)
        ret = self.dense0(state)
        ret = self.dense1(ret)
        ret = self.dense2(ret)
        if self._output_batchnorm:
            ret = self.output_batchnorm(ret, training=training)
        return ret

    def _set_trainable_weights(self):
        self._trainable_weights = []
        self._trainable_weights += self.dense0.trainable_variables
        self._trainable_weights += self.dense1.trainable_variables
        self._trainable_weights += self.dense2.trainable_variables

    def _set_batchnorm_op(self):
        self.batchnorm_op = []
        self.batchnorm_vars = []
        if self._input_batchnorm:
            self.batchnorm_op.append(self.input_batchnorm.updates)
            self.batchnorm_vars += self.input_batchnorm.variables
        if self._output_batchnorm:
            self.batchnorm_op.append(self.output_batchnorm.updates)
            self.batchnorm_vars += self.output_batchnorm.variables


class SimpleCriticModel(MultiInputModel):
    def __init__(self, l0_size, l1_size, input_batchnorm=False, name="critic"):
        super().__init__(name=name)
        self.concat0 = layers.Concatenate()
        self._input_batchnorm = input_batchnorm
        if input_batchnorm:
            self.input_batchnorm = tf.layers.BatchNormalization(scale=False, center=False)
        self.concat1 = layers.Concatenate()
        self.dense0 = layers.Dense(l0_size, activation=tf.tanh, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        self.dense1 = layers.Dense(l1_size, activation=tf.tanh, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        self.dense2 = layers.Dense(1, activation=None)

    def my_call(self, state, goal, action, training=True):
        ret = self.concat0([state, goal, action])
        if self._input_batchnorm:
            ret = self.input_batchnorm(ret, training=training)
        ret = self.dense0(ret)
        ret = self.concat1([state, goal, action, ret])
        ret = self.dense1(ret)
        ret = self.dense2(ret)
        return ret

    def _set_trainable_weights(self):
        self._trainable_weights = []
        self._trainable_weights += self.dense0.trainable_variables
        self._trainable_weights += self.dense1.trainable_variables
        self._trainable_weights += self.dense2.trainable_variables

    def _set_batchnorm_op(self):
        self.batchnorm_op = []
        self.batchnorm_vars = []
        if self._input_batchnorm:
            self.batchnorm_op.append(self.input_batchnorm.updates)
            self.batchnorm_vars += self.input_batchnorm.variables


class DebugReadoutModelBase(MultiInputModel):
    def __init__(self, outsize, name):
        super().__init__(name=name)
        self.dense1 = layers.Dense(50, activation=tf.nn.relu)
        self.dense2 = layers.Dense(outsize, activation=None)

    def my_call(self, inp):
        ret = self.dense1(inp)
        ret = self.dense2(ret)
        return ret

    def _set_trainable_weights(self):
        self._trainable_weights = self.dense1.trainable_variables + self.dense2.trainable_variables


class DebugReadoutGoalModel(DebugReadoutModelBase):
    def __init__(self):
        super().__init__(8, "readout_goal")


class DebugReadoutStateModel(DebugReadoutModelBase):
    def __init__(self):
        super().__init__(12, "readout_state")


class DebugReadoutGStateModel(DebugReadoutModelBase):
    def __init__(self):
        super().__init__(8, "readout_gstate")


class AgencyRootModel(MultiInputModel):
    _indent_level = -1

    def __init__(self, name, childs=[]):
        super().__init__(name=name)
        self.childs = childs

    @staticmethod
    def from_conf(conf_path):
        with open(conf_path, "r") as f:
            conf = json.load(f, cls=AgencyConfDecoder)
        name, conf = conf.popitem()
        return AgencyRootModel.__recinit__(name, conf, is_root=True)

    @staticmethod
    def __recinit__(name, conf, is_root=False):
        childs = [AgencyRootModel.__recinit__(sub_agent_name, sub_agent_conf)
                  for sub_agent_name, sub_agent_conf in conf["childs"].items()]
        if is_root:
            return AgencyRootModel(name, childs=childs)
        else:
            return AgencyModel(conf["policy_model"], conf["target_policy_model"],
                               conf["state_model"],
                               conf["critic_0_model"], conf["target_critic_0_model"],
                               conf["critic_1_model"], conf["target_critic_1_model"],
                               conf["readout_goal_model"], conf["readout_state_model"], conf["readout_gstate_model"],
                               name, childs=childs)

    def __getitem__(self, args):
        if isinstance(args, int):
            return self.childs[args]
        if len(args) == 1:
            return self.childs[args[0]]
        return self.childs[args[0]][args[1:]]

    def browser(self):
        yield self
        for c in self.childs:
            c.width_browser()

    def tree_map(self, func, as_list=False, exclude_root=True):
        if as_list:
            ret = [] if exclude_root else [func(self)]
            return ret + sum((c.tree_map(func, True) for c in self.childs), [])
        else:
            return {self.name: None if exclude_root else func(self),
                    "childs" : [c.tree_map(func, False) for c in self.childs]}

    def my_call(self, parent_goal_0, parent_state_0, parent_gstate_0,
                      parent_goal_1, parent_state_1, parent_gstate_1,
                      learning_rate=None, actor_speed_ratio=None, discount_factor=None, tau=None,
                      behaviour_noise_scale=None, target_smoothing_noise_scale=None,
                      batchnorm_training=True):
        with self._indented_print() as iprint:
            # iprint("calling {} (ROOT)".format(self.name))
            ### CALL SUB-AGENTS
            childs = [child(
                        parent_goal_0, parent_state_0, parent_gstate_0,
                        parent_goal_1, parent_state_1, parent_gstate_1,
                        parent_goal_0, parent_state_0, parent_gstate_0,
                        learning_rate=learning_rate, actor_speed_ratio=actor_speed_ratio, discount_factor=discount_factor, tau=tau,
                        behaviour_noise_scale=behaviour_noise_scale,
                        target_smoothing_noise_scale=target_smoothing_noise_scale,
                        batchnorm_training=batchnorm_training)
                     for child in self.childs]
            ret = AgencyCallRoot(name=self.name, learning_rate=learning_rate, childs=childs)
            # iprint("... done")
        return ret

    def _set_trainable_weights(self):
        self._trainable_weights = self.tree_map(lambda agency: agency.trainable_variables, as_list=True)
        self._trainable_weights = [item for sublist in self._trainable_weights for item in sublist]

    class _indented_print:
        def __enter__(self):
            AgencyRootModel._indent_level += 1
            def print_func(*args, **kwargs):
                print("    " * AgencyRootModel._indent_level, *args, **kwargs)
            return print_func

        def __exit__(self, type, value, traceback):
            AgencyRootModel._indent_level -= 1



class AgencyModel(AgencyRootModel):
    def __init__(self, policy_model, target_policy_model,
                       state_model,
                       critic_0_model, target_critic_0_model,
                       critic_1_model, target_critic_1_model,
                       readout_goal_model, readout_state_model, readout_gstate_model,
                       name, childs=[]):
        super().__init__(name, childs=childs)
        self.policy_model = policy_model
        self.target_policy_model = target_policy_model
        self.state_model = state_model
        self.critic_0_model = critic_0_model
        self.target_critic_0_model = target_critic_0_model
        self.critic_1_model = critic_1_model
        self.target_critic_1_model = target_critic_1_model
        self.readout_goal_model = readout_goal_model
        self.readout_state_model = readout_state_model
        self.readout_gstate_model = readout_gstate_model


    def my_call(self, parent_goal_0, parent_state_0, parent_gstate_0,
                      parent_goal_1, parent_state_1, parent_gstate_1,
                      root_goal, root_state, root_gstate,
                      learning_rate=None, actor_speed_ratio=None, discount_factor=None, tau=None,
                      behaviour_noise_scale=None, target_smoothing_noise_scale=None,
                      batchnorm_training=True):
        with self._indented_print() as iprint:
            # needed if reward computation on the target side
            # tgoal_0 = self.target_policy_model(parent_state_0, tf.stop_gradient(parent_goal_0)) + some_noise?
            # tgstate_0 = self.target_policy_model(parent_state_0, parent_gstate_0) #### warning!! what should be passed to childs?? target or not??
            # tgstate_1 = self.target_policy_model(parent_state_1, parent_gstate_1)
            # reward = d(tgoal_0, tgstate_0) - d(tgoal_0, tgstate_1)
            # iprint("calling {}".format(self.name))
            tensors = FixKerasIssue()
            tensors["name"] = self.name
            ### PARENTS
            tensors["parent_goal_0"] = parent_goal_0
            tensors["parent_state_0"] = parent_state_0
            tensors["parent_gstate_0"] = parent_gstate_0
            tensors["parent_goal_1"] = parent_goal_1
            tensors["parent_state_1"] = parent_state_1
            tensors["parent_gstate_1"] = parent_gstate_1
            ### GOAL
            tensors["goal_0"] = self.policy_model(
                parent_state_0,
                parent_goal_0,
                training=False
            )
            if behaviour_noise_scale:
                tensors["goal_0"] += tf.random_normal(shape=tf.shape(tensors["goal_0"]), stddev=behaviour_noise_scale)
            tensors["goal_1"] = self.policy_model(
                parent_state_1,
                parent_goal_0,  # assumes that the goal has not changed
                training=False
            )
            tensors["goal_0_target"] = self.target_policy_model(
                parent_state_0,
                parent_goal_0,
                training=False
            )
            tensors["goal_1_target"] = self.target_policy_model(
                parent_state_1,
                parent_goal_0,  # assumes that the goal has not changed
                training=False
            )
            if target_smoothing_noise_scale:
                tensors["goal_1_target"] += tf.truncated_normal(shape=tf.shape(tensors["goal_1_target"]), stddev=target_smoothing_noise_scale)
            ### STATE
            tensors["state_0"] = self.state_model(parent_state_0, training=batchnorm_training)
            tensors["state_1"] = self.state_model(parent_state_1, training=False)
            tensors["state_model_variables"] = self.state_model.trainable_variables
            ### GSTATE
            tensors["gstate_0"] = self.policy_model(
                parent_state_0,
                parent_gstate_0,
                training=batchnorm_training
            )
            tensors["gstate_1"] = self.policy_model(
                parent_state_1,
                parent_gstate_1,
                training=False
            )
            tensors["gstate_0_target"] = self.target_policy_model(
                parent_state_0,
                parent_gstate_0,
                training=False
            )
            tensors["gstate_1_target"] = self.target_policy_model(
                parent_state_1,
                parent_gstate_1,
                training=False
            )
            ### STOP GRADIENTS / PLACEHOLDERS
            tensors["placeholder_goal_0"] = tf.placeholder_with_default(
                # tf.stop_gradient(tensors["goal_0"]),
                tensors["goal_0"],
                shape=tensors["goal_0"].get_shape()
            )
            tensors["placeholder_state_0"] = tf.placeholder_with_default(
                # tf.stop_gradient(tensors["state_0"]),
                tensors["state_0"],
                shape=tensors["state_0"].get_shape()
            )
            tensors["placeholder_gstate_0"] = tf.placeholder_with_default(
                # tf.stop_gradient(tensors["gstate_0"]),
                tensors["gstate_0"],
                shape=tensors["gstate_0"].get_shape()
            )
            # apparently useless...
            tensors["placeholder_goal_1"] = tf.placeholder_with_default(
                # tf.stop_gradient(tensors["goal_1"]),
                tensors["goal_1"],
                shape=tensors["goal_1"].get_shape()
            )
            tensors["placeholder_state_1"] = tf.placeholder_with_default(
                # tf.stop_gradient(tensors["state_1"]),
                tensors["state_1"],
                shape=tensors["state_1"].get_shape()
            )
            tensors["placeholder_gstate_1"] = tf.placeholder_with_default(
                # tf.stop_gradient(tensors["gstate_1"]),
                tensors["gstate_1"],
                shape=tensors["gstate_1"].get_shape()
            )
            ### REWARD
            tensors["reward"] = tf.reshape(d(parent_goal_0, parent_gstate_0) - d(parent_goal_0, parent_gstate_1), (-1, 1)) * np.sqrt(1 - discount_factor ** 2)
            ### CRITIC DEFINITION
            # CRITIC 0
            tensors["predicted_return_00"] = self.critic_0_model(
                parent_state_0,
                parent_goal_0,
                tensors["goal_0"],
                training=batchnorm_training
            )
            tensors["predicted_return_10"] = self.critic_0_model(
                parent_state_1,
                parent_goal_0,  # assumes that the goal has not changed
                tensors["goal_1"],  # assumes that the goal has not changed
                training=False
            )
            tensors["predicted_return_00_target"] = self.target_critic_0_model(
                parent_state_0,
                parent_goal_0,
                tensors["goal_0_target"],
                training=False
            )
            tensors["predicted_return_10_target"] = self.target_critic_0_model(
                parent_state_1,
                parent_goal_0,  # assumes that the goal has not changed
                tensors["goal_1_target"],  # assumes that the goal has not changed
                training=False
            )
            # CRITIC 1
            tensors["predicted_return_01"] = self.critic_1_model(
                parent_state_0,
                parent_goal_0,
                tensors["goal_0"],
                training=batchnorm_training
            )
            tensors["predicted_return_11"] = self.critic_1_model(
                parent_state_1,
                parent_goal_0,  # assumes that the goal has not changed
                tensors["goal_1"],  # assumes that the goal has not changed
                training=False
            )
            tensors["predicted_return_01_target"] = self.target_critic_1_model(
                parent_state_0,
                parent_goal_0,
                tensors["goal_0_target"],
                training=False
            )
            tensors["predicted_return_11_target"] = self.target_critic_1_model(
                parent_state_1,
                parent_goal_0,  # assumes that the goal has not changed
                tensors["goal_1_target"],  # assumes that the goal has not changed
                training=False
            )
            # BOTH CRITICS
            tensors["predicted_return_0_target"] = tf.minimum(
                tensors["predicted_return_00_target"],
                tensors["predicted_return_01_target"]
            )
            tensors["predicted_return_1_target"] = tf.minimum(
                tensors["predicted_return_10_target"],
                tensors["predicted_return_11_target"]
            )
            ### CRITIC LOSS
            tensors["critic_target"] = tf.stop_gradient(
                tensors["reward"] + discount_factor * tensors["predicted_return_1_target"]
            )
            # MSE loss
            tensors["critic_0_loss"] = (tensors["predicted_return_00"] - tensors["critic_target"]) ** 2
            tensors["critic_1_loss"] = (tensors["predicted_return_01"] - tensors["critic_target"]) ** 2
            # Huber loss
            # tensors["critic_0_loss"] = tf.losses.huber_loss(tensors["critic_target"], tensors["predicted_return_00"], delta=0.5)
            # tensors["critic_1_loss"] = tf.losses.huber_loss(tensors["critic_target"], tensors["predicted_return_01"], delta=0.5)
            tensors["critic_loss"] = tensors["critic_0_loss"] + tensors["critic_1_loss"]
            tensors["critic_train_op"] = tf.train.AdamOptimizer(learning_rate).minimize(
                tensors["critic_loss"],
                var_list=self.critic_0_model.trainable_variables + self.critic_1_model.trainable_variables
            )
            ### ACTOR LOSS
            # tensors["critic_gradient"] = tf.gradients(tensors["predicted_return_00"], tensors["goal_0"])[0]
            # tensors["actor_train_op"] = tf.train.AdamOptimizer(learning_rate / actor_speed_ratio).minimize(
            #     tensors["goal_0"],
            #     grad_loss=-tensors["critic_gradient"],
            #     var_list=self.policy_model.trainable_variables
            # )
            tensors["actor_train_op"] = tf.train.AdamOptimizer(learning_rate / actor_speed_ratio).minimize(
                -tf.reduce_mean(tensors["predicted_return_00"], axis=0),
                var_list=self.policy_model.trainable_variables
            )
            tensors["batchnorm_train_op"] = \
                self.policy_model.batchnorm_op + \
                self.critic_0_model.batchnorm_op + \
                self.critic_1_model.batchnorm_op + \
                self.state_model.batchnorm_op
            ### TARGET NETWORKS
            # init
            init_target_weights_ops = []
            for w, wtarget in zip(self.policy_model.trainable_variables, self.target_policy_model.trainable_variables):
                init_target_weights_ops.append(wtarget.assign(w))
            for w, wtarget in zip(self.critic_0_model.trainable_variables, self.target_critic_0_model.trainable_variables):
                init_target_weights_ops.append(wtarget.assign(w))
            for w, wtarget in zip(self.critic_1_model.trainable_variables, self.target_critic_1_model.trainable_variables):
                init_target_weights_ops.append(wtarget.assign(w))
            for w, wtarget in zip(self.policy_model.batchnorm_vars, self.target_policy_model.batchnorm_vars):
                init_target_weights_ops.append(wtarget.assign(w))
            for w, wtarget in zip(self.critic_0_model.batchnorm_vars, self.target_critic_0_model.batchnorm_vars):
                init_target_weights_ops.append(wtarget.assign(w))
            for w, wtarget in zip(self.critic_1_model.batchnorm_vars, self.target_critic_1_model.batchnorm_vars):
                init_target_weights_ops.append(wtarget.assign(w))
            tensors["init_target_weights_op"] = tf.group(init_target_weights_ops)
            # update
            update_target_weights_ops = []
            for w, wtarget in zip(self.policy_model.trainable_variables, self.target_policy_model.trainable_variables):
                update_target_weights_ops.append(wtarget.assign(tau * w + (1 - tau) * wtarget))
            for w, wtarget in zip(self.critic_0_model.trainable_variables, self.target_critic_0_model.trainable_variables):
                update_target_weights_ops.append(wtarget.assign(tau * w + (1 - tau) * wtarget))
            for w, wtarget in zip(self.critic_1_model.trainable_variables, self.target_critic_1_model.trainable_variables):
                update_target_weights_ops.append(wtarget.assign(tau * w + (1 - tau) * wtarget))
            for w, wtarget in zip(self.policy_model.batchnorm_vars, self.target_policy_model.batchnorm_vars):
                update_target_weights_ops.append(wtarget.assign(w))
            for w, wtarget in zip(self.critic_0_model.batchnorm_vars, self.target_critic_0_model.batchnorm_vars):
                update_target_weights_ops.append(wtarget.assign(w))
            for w, wtarget in zip(self.critic_1_model.batchnorm_vars, self.target_critic_1_model.batchnorm_vars):
                update_target_weights_ops.append(wtarget.assign(w))
            tensors["update_target_weights_op"] = tf.group(update_target_weights_ops)
            ### PROFILING
            tensors["readout_goal"] = self.readout_goal_model(tf.stop_gradient(tensors["goal_0"]))
            tensors["readout_state"] = self.readout_state_model(tf.stop_gradient(tensors["state_0"]))
            tensors["readout_gstate"] = self.readout_gstate_model(tf.stop_gradient(tensors["gstate_0"]))
            tensors["readout_goal_loss"] = (tensors["readout_goal"] - root_goal) ** 2
            tensors["readout_state_loss"] = (tensors["readout_state"] - root_state) ** 2
            tensors["readout_gstate_loss"] = (tensors["readout_gstate"] - root_gstate) ** 2
            tensors["readout_loss"] = \
                tf.reduce_sum(tensors["readout_goal_loss"]) +\
                tf.reduce_sum(tensors["readout_state_loss"]) +\
                tf.reduce_sum(tensors["readout_gstate_loss"])
            readout_optimizer = tf.train.AdamOptimizer(learning_rate)
            tensors["readout_train_op"] = readout_optimizer.minimize(tensors["readout_loss"])
            ### SUMMARIES
            with tf.name_scope(""):
                # critics
                mean_critic_1_loss_summary = tf.summary.scalar(
                    "/critic_0_loss_mean/{}".format(self.name),
                    tf.reduce_mean(tensors["critic_0_loss"])
                )
                mean_critic_2_loss_summary = tf.summary.scalar(
                    "/critic_1_loss_mean/{}".format(self.name),
                    tf.reduce_mean(tensors["critic_1_loss"])
                )
                mean, var = tf.nn.moments(tensors["reward"], axes=[0, 1])
                std = tf.sqrt(var)
                mean_reward_summary = tf.summary.scalar(
                    "/reward_mean/{}".format(self.name),
                    mean
                )
                std_reward_summary = tf.summary.scalar(
                    "/reward_std/{}".format(self.name),
                    std
                )
                min_reward_summary = tf.summary.scalar(
                    "/reward_min/{}".format(self.name),
                    tf.reduce_min(tensors["reward"])
                )
                max_reward_summary = tf.summary.scalar(
                    "/reward_max/{}".format(self.name),
                    tf.reduce_max(tensors["reward"])
                )
                mean, var = tf.nn.moments(tensors["goal_0"], axes=[0, 1])
                std = tf.sqrt(var)
                mean_goal = tf.summary.scalar(
                    "/goal_mean/{}".format(self.name),
                    mean
                )
                std_goal = tf.summary.scalar(
                    "/goal_std/{}".format(self.name),
                    std
                )
                min_goal = tf.summary.scalar(
                    "/goal_min/{}".format(self.name),
                    tf.reduce_min(tensors["goal_0"])
                )
                max_goal = tf.summary.scalar(
                    "/goal_max/{}".format(self.name),
                    tf.reduce_max(tensors["goal_0"])
                )
                # activity
                goal_histogram = tf.summary.histogram(
                    "/goal/{}".format(self.name),
                    tensors["goal_0"]
                )
                # weights
                critic_1_histograms = [
                    tf.summary.histogram(
                        "/critic_0_{}/{}".format(self.name, "/".join(w.name.split('/')[-2:])),
                        w
                    ) for w in self.critic_0_model.trainable_variables
                ]
                critic_2_histograms = [
                    tf.summary.histogram(
                        "/critic_1_{}/{}".format(self.name, "/".join(w.name.split('/')[-2:])),
                        w
                    ) for w in self.critic_1_model.trainable_variables
                ]
                actor_histogram = [
                    tf.summary.histogram(
                        "/actor_{}/{}".format(self.name, "/".join(w.name.split('/')[-2:])),
                        w
                    ) for w in self.policy_model.trainable_variables
                ]
                state_histogram = [
                    tf.summary.histogram(
                        "/state_{}/{}".format(self.name, "/".join(w.name.split('/')[-2:])),
                        w
                    ) for w in self.state_model.trainable_variables
                ]
                tensors["summary"] = tf.summary.merge([
                    mean_critic_1_loss_summary, mean_critic_2_loss_summary,
                    mean_reward_summary, goal_histogram, std_reward_summary,
                    min_reward_summary, max_reward_summary, mean_goal, std_goal,
                    min_goal, max_goal
                    ] +
                    critic_1_histograms + critic_2_histograms + actor_histogram + state_histogram
                )
            ### CALL SUB-AGENTS
            tensors["childs"] = [child(
                tensors["placeholder_goal_0"], tensors["placeholder_state_0"], tensors["placeholder_gstate_0"],
                tensors["placeholder_goal_1"], tensors["placeholder_state_1"], tensors["placeholder_gstate_1"],
                root_goal, root_state, root_gstate,
                learning_rate=learning_rate, actor_speed_ratio=actor_speed_ratio, discount_factor=discount_factor, tau=tau,
                behaviour_noise_scale=behaviour_noise_scale,
                target_smoothing_noise_scale=target_smoothing_noise_scale)
                for child in self.childs]
            return tensors

    def tree_map(self, func, as_list=False):
        if as_list:
            return [func(self)] + sum((c.tree_map(func, True) for c in self.childs), [])
        else:
            return {self.name: func(self),
                    "childs" : [c.tree_map(func, False) for c in self.childs]}

    def _set_trainable_weights(self):
        self._trainable_weights = []
        self._trainable_weights += self.critic_0_model.trainable_variables
        self._trainable_weights += self.critic_1_model.trainable_variables
        self._trainable_weights += self.target_critic_0_model.trainable_variables
        self._trainable_weights += self.target_critic_1_model.trainable_variables
        self._trainable_weights += self.policy_model.trainable_variables
        self._trainable_weights += self.target_policy_model.trainable_variables
        self._trainable_weights += self.state_model.trainable_variables


class AgencyCallRoot:
    def __init__(self, name, learning_rate, childs=[]):
        self.name = name
        self.childs = [AgencyCall(child) for child in childs]
        ### ACTOR
        self.root_actor_train_op = tf.group(self.tree_map(lambda agency_call: agency_call.actor_train_op, as_list=True))
        self.root_batchnorm_train_op = tf.group(self.tree_map(lambda agency_call: agency_call.batchnorm_train_op, as_list=True))
        ### CRITIC
        self.root_critic_train_op = tf.group(self.tree_map(lambda agency_call: agency_call.critic_train_op, as_list=True))
        ### TARGETS
        self.root_update_target_train_op = tf.group(self.tree_map(lambda agency_call: agency_call.update_target_weights_op, as_list=True))
        self.root_init_target_op = tf.group(self.tree_map(lambda agency_call: agency_call.init_target_weights_op, as_list=True))
        ### STATE
        states_params = self.tree_map(lambda agency: agency.state_model_variables, as_list=True)
        states_params = [item for sublist in states_params for item in sublist]
        # state_loss = sum(self.tree_map(
        #         lambda agency_call:
        #             tf.reduce_mean(
        #             tf.reduce_sum(
        #                 sum([tf.gradients(child.critic_loss, child.parent_state_0)[0] for child in agency_call.childs]) * agency_call.state_0,
        #             axis=-1)),
        #     as_list=True))
        # state_loss = sum(self.tree_map(
        #         lambda agency_call:
        #             tf.reduce_mean(
        #             tf.reduce_sum(
        #                 tf.gradients(agency_call.critic_loss, agency_call.parent_state_0)[0] * agency_call.parent_state_0,
        #             axis=-1)),
        #     as_list=True))
        state_loss = sum(self.tree_map(lambda agency_call: agency_call.critic_loss, as_list=True))
        # state_loss = - sum(self.tree_map(lambda agency_call: agency_call.predicted_return_00, as_list=True))
        state_optimizer = tf.train.AdamOptimizer(learning_rate)
        try:
            self.root_state_train_op = state_optimizer.minimize(state_loss, var_list=states_params)
        except ValueError as err:
            print(err)
            print("The gradient for the state representation could not be computed.\n"
                   "This is a normal behaviour if the agency has a single agent.\n"
                   "Setting the state train op to a tf.no_op!")
            self.root_state_train_op = tf.no_op()
        ### READOUT
        self.root_readout_train_op = tf.group(self.tree_map(lambda agency_call: agency_call.readout_train_op, as_list=True))
        ### SUMMARIES
        self.root_summary_op = tf.summary.merge(self.tree_map(lambda agency_call: agency_call.summary, as_list=True))

    def __getitem__(self, args):
        if isinstance(args, int):
            return self.childs[args]
        if isinstance(args, list):
            if len(args) == 1:
                return self[args[0]]
            return self[args[0]][args[1:]]
        if isinstance(args, str):
            matching = [c for c in self.childs if c.name == args]
            if len(matching) == 1:
                return matching[0]
            if len(matching) > 1:
                raise ValueError("{} is present {} times in the agency call '{}'".format(args, len(matching), self.name))
            if len(matching) == 0:
                possible = set([c.name for c in self.childs])
                raise ValueError("{} could not be found in agency call '{}'. Possible values are: {}".format(args, self.name, possible))

    def tree_map(self, func, as_list=False, exclude_root=True):
        if as_list:
            ret = [] if exclude_root else [func(self)]
            return ret + sum((c.tree_map(func, True) for c in self.childs), [])
        else:
            return {self.name: None if exclude_root else func(self),
                    "childs" : [c.tree_map(func, False) for c in self.childs]}
            # might be better to represent the data this way:
            # return {"name": self.name,
            #         "result": None if exclude_root else func(self),
            #         "childs" : [c.tree_map(func, False) for c in self.childs]}


class AgencyCall(AgencyCallRoot):
    def __init__(self, tensors):
        self.tensors = tensors
        self.childs = [AgencyCall(c) for c in self.tensors.pop("childs")]

    def __getattribute__(self, attribute_name):
        try:
            return object.__getattribute__(self, attribute_name)
        except AttributeError:
            if attribute_name in self.tensors:
                return self.tensors[attribute_name]
            else:
                raise AttributeError("'AgencyCall' object has no attribute {} and {} is not a key in it's 'tensors' dict".format(attribute_name, attribute_name))

    def tree_map(self, func, as_list=False):
        if as_list:
            return [func(self)] + sum((c.tree_map(func, True) for c in self.childs), [])
        else:
            return {self.name: func(self),
                    "childs" : [c.tree_map(func, False) for c in self.childs]}
            # might be better to represent the data this way:
            # return {"name": self.name,
            #         "result": func(self),
            #         "childs" : [c.tree_map(func, False) for c in self.childs]}


if __name__ == "__main__":
    # a = AgencyRootModel.from_conf("../agencies/left_elbow_agency.txt")
    a = AgencyRootModel.from_conf("../agencies/simple_agency.txt")
    # a = AgencyRootModel.from_conf("../agencies/debug_agency.txt")

    parent_goal_0 = tf.placeholder(shape=(None, 8), dtype=tf.float32)
    parent_state_0 = tf.placeholder(shape=(None, 12), dtype=tf.float32)
    parent_gstate_0 = tf.placeholder(shape=(None, 8), dtype=tf.float32)
    parent_goal_1 = tf.placeholder(shape=(None, 8), dtype=tf.float32)
    parent_state_1 = tf.placeholder(shape=(None, 12), dtype=tf.float32)
    parent_gstate_1 = tf.placeholder(shape=(None, 8), dtype=tf.float32)

    b = a(
        parent_goal_0,
        parent_state_0,
        parent_gstate_0,
        parent_goal_1,
        parent_state_1,
        parent_gstate_1,
        learning_rate=1e-3,
        discount_factor=0.85,
        tau=5e-3,
        behaviour_noise_scale=0.05,
        target_smoothing_noise_scale=0.002
    )
