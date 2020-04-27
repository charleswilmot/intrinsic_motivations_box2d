import time
from imageio import get_writer
import sys
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
from qimage2ndarray import rgb_view
import tensorflow as tf


class EnvironmentScene(QtGui.QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        # self.refresh_data()

    def refresh_data(self, ploting_data):
        if ploting_data is not None:
            self.clear()
            for key, data in ploting_data.items():
                polygon = QtGui.QPolygonF()
                for p in data:
                    polygon.append(QtCore.QPointF(*p))
                self.addPolygon(polygon)


# remove env from the class, pass x y w h to the contructor
class EnvironmentWidget(QtGui.QGraphicsView):
    def __init__(self, x0, x1, y0, y1, parent=None):
        super().__init__(parent=parent)
        self._scene = EnvironmentScene(parent=self)
        self.setScene(self._scene)
        self._x = x0
        self._y = y0
        self._w = x1 - x0
        self._h = y1 - y0
        self.fitInView()

    def fitInView(self):
        super().fitInView(self._x, self._y, self._w, self._h, 1)

    def resizeEvent(self, e):
        self.fitInView()

    def refresh_data(self, ploting_data):
        self._scene.refresh_data(ploting_data)


class OneValueWidget(pg.PlotWidget):
    def __init__(self, color=(255, 128, 128), lookback=50, parent=None):
        super().__init__(parent=parent)
        self._x = np.arange(-lookback + 1, 1)
        self._value_buffer = np.zeros(lookback)
        self.setWindowTitle('reward')
        self.setRange(QtCore.QRectF(-lookback, -1, lookback, 2))
        self.setLabel('bottom', 'iterations')
        self._reward_curve = self.plot(pen=pg.mkPen(*color))

    def refresh_data(self, value):
        self._value_buffer[:-1] = self._value_buffer[1:]
        self._value_buffer[-1] = value
        self._reward_curve.setData(x=self._x, y=self._value_buffer)



class ReturnWidget(pg.PlotWidget):
    def __init__(self, name, discount_factor, lookback=50, parent=None):
        super().__init__(parent=parent)
        self._x = np.arange(-lookback + 1, 1)
        self._return_buffer = np.zeros(lookback)
        self._prediction_buffer = np.zeros(lookback)
        self._target_buffer = np.zeros(lookback)
        self._prediction_part = np.zeros(lookback)
        self._gammas = np.cumprod(np.full(lookback, discount_factor))[::-1] / discount_factor
        self._discount_factor = discount_factor
        # self.setWindowTitle(name)
        self.setRange(QtCore.QRectF(-lookback, -1, lookback, 2))
        self.setLabel('bottom', name)
        self._return_curve = self.plot(pen=pg.mkPen(255, 255, 0))
        self._prediction_curve = self.plot(pen=pg.mkPen(51, 153, 255))
        self._target_curve = self.plot(pen=pg.mkPen(126, 247, 130))

    def refresh_data(self, reward, prediction, target):
        self._prediction_buffer[:-1] = self._prediction_buffer[1:]
        self._prediction_buffer[-1] = prediction
        self._target_buffer[:-1] = self._target_buffer[1:]
        self._target_buffer[-1] = target
        self._return_buffer[:-1] = self._return_buffer[1:]
        self._return_buffer += self._gammas * reward - self._prediction_part
        self._prediction_part = prediction * self._gammas
        self._return_buffer[-1] = reward
        self._return_buffer += self._discount_factor * self._prediction_part
        self._return_curve.setData(x=self._x, y=self._return_buffer)
        self._prediction_curve.setData(x=self._x, y=self._prediction_buffer)
        self._target_curve.setData(x=self._x, y=self._target_buffer)


class NonLeafAgentWidget(QtGui.QWidget):
    def __init__(self, name, discount_factor, childs_names, level, parent=None):
        super().__init__(parent=parent)
        self._name_widget = QtGui.QLabel(name)
        lookback = 50 * 10 ** level
        self._reward = OneValueWidget(lookback=lookback, parent=self)
        self._distance = OneValueWidget((0, 255, 0), lookback=lookback, parent=self)
        self._distance.setYRange(-0.1, 2)
        self._returns = [ReturnWidget(child_name, discount_factor, lookback=lookback, parent=self) for child_name in childs_names]
        self._layout = QtGui.QGridLayout(self)
        self._layout.addWidget(self._name_widget, 0, 0, 1, 1)
        self._layout.addWidget(self._distance, 1, 0, 1, 1)
        self._layout.addWidget(self._reward, 2, 0, 1, 1)
        for i, _return in enumerate(self._returns):
            self._layout.addWidget(_return, 3 + i, 0, 1, 1)
        self._layout.setSpacing(0)
        self._layout.setMargin(0)
        self.setLayout(self._layout)

    def refresh_data(self, distance, reward, predicted_returns, targets):
        self._distance.refresh_data(distance)
        self._reward.refresh_data(reward)
        for predicted_return, target, _return in zip(predicted_returns, targets, self._returns):
            _return.refresh_data(reward, predicted_return, target)


class Window(QtGui.QMainWindow):
    def __init__(self, env, child_names, discount_factor, save=False, save_path=None):
        super().__init__()
        # self.setGeometry(50, 50, 1000, 600)
        self._env = env
        self._discount_factor = discount_factor
        self._central_widget = QtGui.QWidget(parent=self)
        self.setCentralWidget(self._central_widget)
        self._goal_0_widget = EnvironmentWidget(*env.renderer._x_lim, *env.renderer._y_lim, self._central_widget)
        self._goal_1_widget = EnvironmentWidget(*env.renderer._x_lim, *env.renderer._y_lim, self._central_widget)
        self._environment_widget = EnvironmentWidget(*env.renderer._x_lim, *env.renderer._y_lim, self._central_widget)
        self._agents = self._create_agents_widget(child_names)
        self._layout = QtGui.QGridLayout(self._central_widget)
        self._layout.addWidget(self._goal_0_widget, 0, 0, 1, 1)
        self._layout.addWidget(self._environment_widget, 1, 0, 1, 1)
        self._layout.addWidget(self._goal_1_widget, 2, 0, 1, 1)
        self._last_level = len(self._agents)
        self._save = save
        self._saved = 0
        self._writer = get_writer(save_path, fps=25) if self._save and save_path is not None else None
        i = 1
        for level in self._agents:
            for agent in level:
                self._layout.addWidget(agent, 0, i, 4, 1)
                i += 1

    def _create_agents_widget(self, child_names, level=0):
        # names is a dict tree like so:
        # {"all_body": ["child1", "child2"], "childs": [{"child1": [...], "childs": [...]}, {"child2": [...], "childs": [...]}]}
        # this function returns a list of list of NonLeafAgentWidget
        if not len(child_names["childs"]):  # has no childs
            return []
        name = [key for key in child_names if key != "childs"][0]
        here = [[NonLeafAgentWidget(name, self._discount_factor, child_names[name], level, parent=self._central_widget)]]
        lower = [self._create_agents_widget(lowwer_lever_child_names, level=level + 1) for lowwer_lever_child_names in child_names["childs"]]
        return here + [sum(level, []) for level in zip(*lower)]

    def _refresh_agents(self, level, data):
        for agent, d in zip(self._agents[level], data):
            agent.refresh_data(
                d["mean_distance_to_goal"],
                d["reward"][0],
                d["predicted_return"][0],
                d["critic_target"][0]
            )

    def refresh_data(self, level, data):
        self._refresh_agents(level, data)
        self._environment_widget.refresh_data(self._env.ploting_data)
        self.save_frame_if_needed(level)

    def save_frame_if_needed(self, level):
        self.update()
        if level == self._last_level - 1 and self._save:
            self._saved += 1
            pixmap = QtGui.QPixmap.grabWindow(self._central_widget.winId())
            image = pixmap.toImage()
            image = image.convertToFormat(QtGui.QImage.Format_RGB32)
            self._writer.append_data(rgb_view(image))
            if self._saved == self._save:
                self._writer.close()
                self.emit(QtCore.SIGNAL("timeToExit"))

    def refresh_goals(self, ploting_data_0, ploting_data_1):
        self._goal_0_widget.refresh_data(ploting_data_0)
        self._goal_1_widget.refresh_data(ploting_data_1)


class ComputationCenter(QtCore.QThread):
    def __init__(self, worker, training=None):
        QtCore.QThread.__init__(self)
        self.worker = worker

        def display_fetches_map_func(agency_call):
            return {
                # "goal_0": agency_call.goal_0,
                "reward": agency_call.reward,
                "mean_distance_to_goal": agency_call.mean_distance_to_goal,
                "predicted_return": agency_call.predicted_return_00,
                "critic_target": agency_call.critic_target
            }

        self.display_fetches = self.worker.agency_test.map_level(display_fetches_map_func)

        self._counts = 0
        self.last_level = self.worker.last_level
        self._display_data = [None] * self.last_level
        self._level_changed = [False] * self.last_level
        if training is None:
            self._training = [False for a in self.display_fetches]
        else:
            self._training = training

    def display_feed_dict(self, level, goals, state_before, gstate_before, state_after, gstate_after):
        # behaviour_goals_placeholders = self.worker.training_behaviour_goals_placeholders[level] \
        #         if self._training[level] else self.worker.testing_behaviour_goals_placeholders[level]
        behaviour_goals_placeholders = self.worker.testing_behaviour_goals_placeholders[level]
        feed_dict = {}
        for placeholder, goal in zip(behaviour_goals_placeholders, goals):
            feed_dict[placeholder] = goal.reshape((1, -1))
        feed_dict[self.worker.parent_state_0] = state_before
        feed_dict[self.worker.parent_gstate_0] = gstate_before
        feed_dict[self.worker.parent_state_1] = state_after
        feed_dict[self.worker.parent_gstate_1] = gstate_after
        return feed_dict

    def refresh_updated_levels(self):
        for level in range(self.last_level):
            if self._level_changed[level]:
                self.emit(QtCore.SIGNAL("refreshData"), level, self._display_data[level])
                self._level_changed[level] = False

    def recursively_apply_action(self, level, goals):
        state_before = self.worker.get_state()
        gstate_before = self.worker.get_gstate()
        state_after = state_before
        gstate_after = gstate_before
        train = self._training[level]
        behaviour_fetches = self.worker.training_behaviour_fetches[level] if train else self.worker.testing_behaviour_fetches[level]
        for i in range(self.worker.time_scale_factor):
            ### BEFORE ###
            state_before = state_after
            gstate_before = gstate_after
            feed_dict = self.worker.behaviour_feed_dict(level, goals, state_before, gstate_before, train=train)
            try:
                transition_0 = self.worker.sess.run(behaviour_fetches, feed_dict=feed_dict)
            except tf.errors.UnavailableError as e:
                return True
            action = transition_0["goal_0"]
            if level + 1 == self.last_level:
                self.worker.apply(action)
                self.worker.register_current_gstate()
                self.refresh_updated_levels()
                must_stop = False
            else:
                must_stop = self.recursively_apply_action(level + 1, action)
            if must_stop:
                return True
            ### AFTER ###
            state_after = self.worker.get_state()
            gstate_after = self.worker.get_gstate()
            feed_dict = self.display_feed_dict(level, goals, state_before, gstate_before, state_after, gstate_after)
            try:
                display_data = self.worker.sess.run(self.display_fetches[level], feed_dict=feed_dict)
            except tf.errors.UnavailableError as e:
                return True
            self._display_data[level] = display_data
            self._level_changed[level] = True
        return False

    def one_high_level_iteration(self, goals):
        return self.recursively_apply_action(0, goals)

    def one_episode(self):
        goals, (ploting_data_0, ploting_data_1) = self.worker.goals_buffer.sample()
        self.emit(QtCore.SIGNAL("refreshGoals"), ploting_data_0, ploting_data_1)
        return self.one_high_level_iteration([goals])

    def check_for_termination(self):
        if self._n_frames_to_render <= 0:
            self.emit(QtCore.SIGNAL("timeToExit"))
            return True
        return False

    def run(self):
        while not self.one_episode():
            pass


# could very well be a method inside worker ? as well as the ComputationCenter ? Could the worker inherit from QThread itself?
class Display:
    def __init__(self, worker, training=None, save=False, save_path=None):
        self.worker = worker
        self._training = training
        self.app = QtGui.QApplication(sys.argv)
        # initialize window
        child_names = self.worker.agency_test.tree_map(lambda agency_call: [c.name for c in agency_call.childs], exclude_root=False)
        self.window = Window(self.worker.env, child_names, self.worker.discount_factor, save=save, save_path=save_path)
        self.window.resize(1536, 864)
        self.window.show()
        self.computation_center = ComputationCenter(worker, training=training)
        QtCore.QObject.connect(self.computation_center, QtCore.SIGNAL("refreshData"), self.window.refresh_data, QtCore.Qt.BlockingQueuedConnection)
        QtCore.QObject.connect(self.computation_center, QtCore.SIGNAL("refreshGoals"), self.window.refresh_goals, QtCore.Qt.BlockingQueuedConnection)
        QtCore.QObject.connect(self.computation_center, QtCore.SIGNAL("timeToExit"), self.app.exit, QtCore.Qt.QueuedConnection)
        QtCore.QObject.connect(self.window, QtCore.SIGNAL("timeToExit"), self.app.exit, QtCore.Qt.QueuedConnection)
        self.computation_center.start()
        self.app.exec_()
