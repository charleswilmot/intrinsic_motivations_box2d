import tensorflow as tf
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import layers
import numpy as np
import json


def d(a, b):
    return tf.reduce_sum((a - b) ** 2, axis=-1)


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
                ret["critic_1_model"] = self._to_model_class(code)
                ret["target_critic_1_model"] = self._to_model_class(code)
                ret["critic_2_model"] = self._to_model_class(code)
                ret["target_critic_2_model"] = self._to_model_class(code)
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
        return self.my_call(*inputs, **kwargs)


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

    def build(self, inputs):
        self._trainable_weights = self.dense.trainable_variables


class DebugStateModel(MultiInputModel):
    def __init__(self, name="state"):
        super().__init__(name=name)
        self.dense = layers.Dense(10)

    def my_call(self, state):
        return self.dense(state)

    def build(self, inputs):
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

    def build(self, inputs):
        self._trainable_weights = self.dense.trainable_variables


class DebugReadoutModelBase(MultiInputModel):
    def __init__(self, outsize, name):
        super().__init__(name=name)
        self.dense1 = layers.Dense(50, activation=tf.nn.relu)
        self.dense2 = layers.Dense(outsize, activation=None)

    def my_call(self, inp):
        ret = tf.stop_gradient(inp)
        ret = self.dense1(inp)
        ret = self.dense2(ret)
        return ret

    def build(self, inputs):
        self._trainable_weights = self.dense1.trainable_variables + self.dense2.trainable_variables


class DebugReadoutGoalModel(DebugReadoutModelBase):
    def __init__(self):
        super().__init__(10, "readout_goal")


class DebugReadoutStateModel(DebugReadoutModelBase):
    def __init__(self):
        super().__init__(10, "readout_state")


class DebugReadoutGStateModel(DebugReadoutModelBase):
    def __init__(self):
        super().__init__(10, "readout_gstate")


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
                               conf["critic_1_model"], conf["target_critic_1_model"],
                               conf["critic_2_model"], conf["target_critic_2_model"],
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
                      learning_rate=None, discount_factor=None, tau=None,
                      behaviour_noise_scale=None, target_smoothing_noise_scale=None):
        with self._indented_print() as iprint:
            iprint("calling {} (ROOT)".format(self.name))
            ### CALL SUB-AGENTS
            childs = [child(
                        parent_goal_0, parent_state_0, parent_gstate_0,
                        parent_goal_1, parent_state_1, parent_gstate_1,
                        parent_goal_0, parent_state_0, parent_gstate_0,
                        learning_rate=learning_rate, discount_factor=discount_factor, tau=tau,
                        behaviour_noise_scale=behaviour_noise_scale,
                        target_smoothing_noise_scale=target_smoothing_noise_scale)
                     for child in self.childs]
            ###### OPERATION OVER ENTIRE TREE
            ret = AgencyCallRoot(name=self.name, childs=childs)
            ### ACTOR
            root_actor_train_op = tf.group(ret.tree_map(lambda agency_call: agency_call.actor_train_op, as_list=True))
            ### CRITIC
            root_critic_train_op = tf.group(ret.tree_map(lambda agency_call: agency_call.critic_train_op, as_list=True))
            ### STATE
            states_params = self.tree_map(lambda agency: agency.state_model.trainable_variables, as_list=True)
            states_params = [item for sublist in states_params for item in sublist]
            state_loss = sum(ret.tree_map(lambda agency_call: agency_call.actor_loss, as_list=True))
            state_optimizer = tf.train.AdamOptimizer(learning_rate)
            root_state_train_op = state_optimizer.minimize(state_loss, var_list=states_params)
            ret.set_root_train_ops(
                root_actor_train_op=root_actor_train_op,
                root_critic_train_op=root_critic_train_op,
                root_state_train_op=root_state_train_op
            )
            iprint("... done")
        return ret

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
                       critic_1_model, target_critic_1_model,
                       critic_2_model, target_critic_2_model,
                       readout_goal_model, readout_state_model, readout_gstate_model,
                       name, childs=[]):
        super().__init__(name, childs=childs)
        self.policy_model = policy_model
        self.target_policy_model = target_policy_model
        self.state_model = state_model
        self.critic_1_model = critic_1_model
        self.target_critic_1_model = target_critic_1_model
        self.critic_2_model = critic_2_model
        self.target_critic_2_model = target_critic_2_model
        self.readout_goal_model = readout_goal_model
        self.readout_state_model = readout_state_model
        self.readout_gstate_model = readout_gstate_model


    def my_call(self, parent_goal_0, parent_state_0, parent_gstate_0,
                      parent_goal_1, parent_state_1, parent_gstate_1,
                      root_goal, root_state, root_gstate,
                      learning_rate=None, discount_factor=None, tau=None,
                      behaviour_noise_scale=None, target_smoothing_noise_scale=None,
                      batchnorm_training=True):
        with self._indented_print() as iprint:
            # needed if reward computation on the target side
            # tgoal_0 = self.target_policy_model(parent_state_0, tf.stop_gradient(parent_goal_0)) + some_noise?
            # tgstate_0 = self.target_policy_model(parent_state_0, parent_gstate_0) #### warning!! what should be passed to childs?? target or not??
            # tgstate_1 = self.target_policy_model(parent_state_1, parent_gstate_1)
            # reward = d(tgoal_0, tgstate_0) - d(tgoal_0, tgstate_1)
            iprint("calling {}".format(self.name))
            ### GOAL
            goal_0 = self.policy_model(parent_state_0, tf.stop_gradient(parent_goal_0), training=batchnorm_training)
            if behaviour_noise_scale:
                goal_0 += tf.random_normal(shape=tf.shape(goal_0), stddev=behaviour_noise_scale)
            goal_1 = self.target_policy_model(parent_state_1, tf.stop_gradient(parent_goal_0), training=batchnorm_training)
            if target_smoothing_noise_scale:
                goal_1 += tf.truncated_normal(shape=tf.shape(goal_1), stddev=target_smoothing_noise_scale)
            her_goal_0 = self.policy_model(parent_state_0, tf.stop_gradient(parent_gstate_1), training=batchnorm_training)
            ### STATE
            state_0 = self.state_model(parent_state_0)
            state_1 = self.state_model(parent_state_1)
            ### STATE IN GOAL SPACE
            gstate_0 = self.policy_model(parent_state_0, parent_gstate_0, training=batchnorm_training)
            gstate_1 = self.policy_model(parent_state_1, parent_gstate_1, training=batchnorm_training)
            ### PREDICTED RETURN
            predicted_return_1 = self.critic_1_model(tf.stop_gradient(parent_state_0), tf.stop_gradient(parent_goal_0), goal_0)
            predicted_return_target_1 = self.target_critic_1_model(parent_state_1, parent_goal_0, goal_1)
            predicted_return_2 = self.critic_2_model(tf.stop_gradient(parent_state_0), tf.stop_gradient(parent_goal_0), goal_0)
            predicted_return_target_2 = self.target_critic_2_model(parent_state_1, parent_goal_0, goal_1)
            predicted_return = tf.minimum(predicted_return_target_1, predicted_return_target_2)
            her_predicted_return_1 = self.critic_1_model(tf.stop_gradient(parent_state_0), tf.stop_gradient(parent_gstate_1), goal_0)
            her_predicted_return_2 = self.critic_2_model(tf.stop_gradient(parent_state_0), tf.stop_gradient(parent_gstate_1), goal_0)
            ### CRITIC
            reward = d(goal_0, gstate_0) - d(goal_0, gstate_1)
            critic_target = reward + discount_factor * predicted_return
            critic_1_loss = (predicted_return_1 - tf.stop_gradient(critic_target)) ** 2
            critic_2_loss = (predicted_return_2 - tf.stop_gradient(critic_target)) ** 2
            her_critic_1_loss = (her_predicted_return_1 - tf.stop_gradient(d(parent_state_0, parent_state_1))) ** 2
            her_critic_2_loss = (her_predicted_return_2 - tf.stop_gradient(d(parent_state_0, parent_state_1))) ** 2
            critic_loss = critic_1_loss + critic_2_loss + her_critic_1_loss + her_critic_2_loss
            critic_optimizer = tf.train.AdamOptimizer(learning_rate)
            critic_train_op = critic_optimizer.minimize(
                critic_loss,
                var_list=self.critic_1_model.trainable_variables + self.critic_2_model.trainable_variables)
            ### ACTOR
            actor_loss = -predicted_return_1
            her_actor_loss = (her_goal_0 - tf.stop_gradient(goal_0)) ** 2
            actor_loss += her_actor_loss
            actor_optimizer = tf.train.AdamOptimizer(learning_rate)
            actor_train_op = actor_optimizer.minimize(actor_loss, var_list=self.policy_model.trainable_variables)
            batchnorm_train_op = self.policy_model.batchnorm.updates + self.target_policy_model.batchnorm.updates
            ### TARGET WEIGHTS UPDATES
            update_target_weights_ops = []
            for w, wtarget in zip(self.policy_model.trainable_variables, self.target_policy_model.trainable_variables):
                update_target_weights_ops.append(wtarget.assign(tau * w + (1 - tau) * wtarget))
            for w, wtarget in zip(self.critic_1_model.trainable_variables, self.target_critic_1_model.trainable_variables):
                update_target_weights_ops.append(wtarget.assign(tau * w + (1 - tau) * wtarget))
            for w, wtarget in zip(self.critic_2_model.trainable_variables, self.target_critic_2_model.trainable_variables):
                update_target_weights_ops.append(wtarget.assign(tau * w + (1 - tau) * wtarget))
            update_target_weights_op = tf.group(update_target_weights_ops)
            ### PROFILING
            readout_goal = self.readout_goal_model(goal_0)
            readout_state = self.readout_state_model(state_0)
            readout_gstate = self.readout_gstate_model(gstate_0)
            readout_goal_loss = (readout_goal - root_goal) ** 2
            readout_state_loss = (readout_state - root_state) ** 2
            readout_gstate_loss = (readout_gstate - root_gstate) ** 2
            readout_loss = \
                tf.reduce_sum(readout_goal_loss) +\
                tf.reduce_sum(readout_state_loss) +\
                tf.reduce_sum(readout_gstate_loss)
            readout_optimizer = tf.train.AdamOptimizer(learning_rate)
            readout_train_op = readout_optimizer.minimize(readout_loss)
            ### SUMMARIES
            mean_actor_loss_summary = tf.summary.scalar("/{}/mean_actor_loss".format(self.name), tf.reduce_mean(actor_loss))
            mean_her_actor_loss_summary = tf.summary.scalar("/{}/mean_her_actor_loss".format(self.name), tf.reduce_mean(her_actor_loss))
            mean_critic_1_loss_summary = tf.summary.scalar("/{}/mean_critic_1_loss".format(self.name), tf.reduce_mean(critic_1_loss))
            mean_critic_2_loss_summary = tf.summary.scalar("/{}/mean_critic_2_loss".format(self.name), tf.reduce_mean(critic_2_loss))
            mean_her_critic_1_loss_summary = tf.summary.scalar("/{}/mean_her_critic_1_loss".format(self.name), tf.reduce_mean(her_critic_1_loss))
            mean_her_critic_2_loss_summary = tf.summary.scalar("/{}/mean_her_critic_2_loss".format(self.name), tf.reduce_mean(her_critic_2_loss))
            mean_reward_summary = tf.summary.scalar("/{}/mean_reward".format(self.name), tf.reduce_mean(reward))
            summary = tf.summary.merge([
                mean_actor_loss_summary, mean_her_actor_loss_summary, mean_critic_1_loss_summary,
                mean_critic_2_loss_summary, mean_her_critic_1_loss_summary, mean_her_critic_2_loss_summary,
                mean_reward_summary])
            ### CALL SUB-AGENTS
            childs = [child(
                        goal_0, state_0, gstate_0,
                        goal_1, state_1, gstate_1,
                        root_goal, root_state, root_gstate,
                        learning_rate=learning_rate, discount_factor=discount_factor, tau=tau,
                        behaviour_noise_scale=behaviour_noise_scale,
                        target_smoothing_noise_scale=target_smoothing_noise_scale)
                     for child in self.childs]
            ret = AgencyCall(
                goal_0=goal_0,
                goal_1=goal_1,
                state_0=state_0,
                state_1=state_1,
                gstate_0=gstate_0,
                gstate_1=gstate_1,
                predicted_return_1=predicted_return_1,
                predicted_return_target_1=predicted_return_target_1,
                predicted_return_2=predicted_return_2,
                predicted_return_target_2=predicted_return_target_2,
                predicted_return=predicted_return,
                reward=reward,
                critic_target=critic_target,
                critic_1_loss=critic_1_loss,
                critic_2_loss=critic_2_loss,
                critic_loss=critic_loss,
                critic_train_op=critic_train_op,
                actor_loss=actor_loss,
                actor_train_op=actor_train_op,
                update_target_weights_op=update_target_weights_op,
                batchnorm_train_op=batchnorm_train_op,
                her_critic_1_loss=her_critic_1_loss,
                her_predicted_return_1=her_predicted_return_1,
                her_critic_2_loss=her_critic_2_loss,
                her_predicted_return_2=her_predicted_return_2,
                her_actor_loss=her_actor_loss,
                her_goal_0=her_goal_0,
                readout_goal=readout_goal,
                readout_state=readout_state,
                readout_gstate=readout_gstate,
                readout_goal_loss=readout_goal_loss,
                readout_state_loss=readout_state_loss,
                readout_gstate_loss=readout_gstate_loss,
                readout_loss=readout_loss,
                readout_train_op=readout_train_op,
                summary=summary,
                name=self.name,
                childs=childs)
            iprint("... done")
        return ret

    def tree_map(self, func, as_list=False):
        if as_list:
            return [func(self)] + sum((c.tree_map(func, True) for c in self.childs), [])
        else:
            return {self.name: func(self),
                    "childs" : [c.tree_map(func, False) for c in self.childs]}


class AgencyCallRoot:
    def __init__(self, name, childs=[]):
        self.name = name
        self.childs = childs

    def __getitem__(self, args):
        if isinstance(args, int):
            return self.childs[args]
        if len(args) == 1:
            return self.childs[args[0]]
        return self.childs[args[0]][args[1:]]

    def set_root_train_ops(self, root_actor_train_op=None, root_critic_train_op=None, root_state_train_op=None):
        self.root_actor_train_op = root_actor_train_op
        self.root_critic_train_op = root_critic_train_op
        self.root_state_train_op = root_state_train_op

    def tree_map(self, func, as_list=False, exclude_root=True):
        if as_list:
            ret = [] if exclude_root else [func(self)]
            return ret + sum((c.tree_map(func, True) for c in self.childs), [])
        else:
            return {self.name: None if exclude_root else func(self),
                    "childs" : [c.tree_map(func, False) for c in self.childs]}


class AgencyCall(AgencyCallRoot):
    def __init__(self,
                 goal_0, goal_1, state_0, state_1, gstate_0, gstate_1,
                 predicted_return_1, predicted_return_target_1, predicted_return_2, predicted_return_target_2, predicted_return,
                 reward, critic_target, critic_1_loss, critic_2_loss, critic_loss, critic_train_op,
                 actor_loss, actor_train_op, update_target_weights_op, batchnorm_train_op,
                 her_critic_1_loss, her_predicted_return_1, her_critic_2_loss, her_predicted_return_2, her_actor_loss, her_goal_0,
                 readout_goal, readout_state, readout_gstate, readout_goal_loss, readout_state_loss, readout_gstate_loss,
                 readout_loss, readout_train_op, summary, name, childs=[]):
        super().__init__(name, childs=childs)
        self.goal_0 = goal_0
        self.goal_1 = goal_1
        self.state_0 = state_0
        self.state_1 = state_1
        self.gstate_0 = gstate_0
        self.gstate_1 = gstate_1
        self.predicted_return_1 = predicted_return_1
        self.predicted_return_target_1 = predicted_return_target_1
        self.predicted_return_2 = predicted_return_2
        self.predicted_return_target_2 = predicted_return_target_2
        self.predicted_return = predicted_return
        self.reward = reward
        self.critic_target = critic_target
        self.critic_1_loss = critic_1_loss
        self.critic_2_loss = critic_2_loss
        self.critic_loss = critic_loss
        self.critic_train_op = critic_train_op
        self.actor_loss = actor_loss
        self.actor_train_op = actor_train_op
        self.update_target_weights_op = update_target_weights_op
        self.batchnorm_train_op = batchnorm_train_op
        self.her_critic_1_loss = her_critic_1_loss
        self.her_predicted_return_1 = her_predicted_return_1
        self.her_critic_2_loss = her_critic_2_loss
        self.her_predicted_return_2 = her_predicted_return_2
        self.her_actor_loss = her_actor_loss
        self.her_goal_0 = her_goal_0
        self.readout_goal = readout_goal
        self.readout_state = readout_state
        self.readout_gstate = readout_gstate
        self.readout_goal_loss = readout_goal_loss
        self.readout_state_loss = readout_state_loss
        self.readout_gstate_loss = readout_gstate_loss
        self.readout_loss = readout_loss
        self.readout_train_op = readout_train_op
        self.summary = summary

    def set_root_train_ops(self, root_actor_train_op=None, root_critic_train_op=None, root_state_train_op=None):
        raise ValueError("This method can only be called on the root of the tree.")

    def tree_map(self, func, as_list=False):
        if as_list:
            return [func(self)] + sum((c.tree_map(func, True) for c in self.childs), [])
        else:
            return {self.name: func(self),
                    "childs" : [c.tree_map(func, False) for c in self.childs]}



if __name__ == "__main__":
    a = AgencyRootModel.from_conf("../agencies/debug_agency.txt")

    parent_goal_0 = tf.placeholder(shape=(None, 10), dtype=tf.float32)
    parent_state_0 = tf.placeholder(shape=(None, 10), dtype=tf.float32)
    parent_gstate_0 = tf.placeholder(shape=(None, 10), dtype=tf.float32)
    parent_goal_1 = tf.placeholder(shape=(None, 10), dtype=tf.float32)
    parent_state_1 = tf.placeholder(shape=(None, 10), dtype=tf.float32)
    parent_gstate_1 = tf.placeholder(shape=(None, 10), dtype=tf.float32)

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
