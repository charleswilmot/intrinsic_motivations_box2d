from imageio import get_writer
import sys
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np


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


# class ReadoutWidget(pg.GraphicsLayoutWidget):
#     def __init__(self, parent=None):
#         super().__init__(parent=parent)
#         self._view = self.addViewBox()
#         self._scatter = pg.ScatterPlotItem(x=np.arange(10), y=np.random.uniform(size=10))
#         self._view.addItem(self._scatter)
#         self._x = None
#
#     def refresh_data(self, new):
#         if self._x is None:
#             self._x = np.arange(len(new))
#         self._scatter.setData(self._x, new) ## have a look at http://www.pyqtgraph.org/documentation/_modules/pyqtgraph/graphicsItems/ScatterPlotItem.html#ScatterPlotItem.addPoints


# class ThreeReadoutsWidget(QtGui.QWidget):
#     def __init__(self, parent=None, horizontal=False):
#         super().__init__(parent=parent)
#         # self.resize(600, 100)
#         self._state = ReadoutWidget(parent=self)
#         self._goal = ReadoutWidget(parent=self)
#         self._gstate = ReadoutWidget(parent=self)
#         self._layout = QtGui.QHBoxLayout(self) if horizontal else QtGui.QVBoxLayout(self)
#         self._layout.addWidget(self._state)
#         self._layout.addWidget(self._goal)
#         self._layout.addWidget(self._gstate)
#         self.setLayout(self._layout)
#         self._layout.setSpacing(0)
#         self._layout.setMargin(0)
#
#     def refresh_data(self, state, goal, gstate):
#         self._state.refresh_data(state)
#         self._goal.refresh_data(goal)
#         self._gstate.refresh_data(gstate)


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


# to update / merge with NonLeafAgentWidget
# class RewardReturnWidget(QtGui.QWidget):
#     def __init__(self, discount_factor, parent=None):
#         super().__init__(parent=parent)
#         self._reward = OneValueWidget(parent=self)
#         self._return = ReturnWidget(discount_factor, parent=self)
#         self._layout = QtGui.QHBoxLayout(self)
#         self._layout.addWidget(self._reward)
#         self._layout.addWidget(self._return)
#         self._layout.setSpacing(0)
#         self._layout.setMargin(0)
#         self.setLayout(self._layout)
#
#     def refresh_data(self, reward, predicted_return, target):
#         self._reward.refresh_data(reward)
#         self._return.refresh_data(reward, predicted_return, target)


# to update / merge with RewardReturnWidget
class NonLeafAgentWidget(QtGui.QWidget):
    def __init__(self, name, discount_factor, childs_names, parent=None):
        super().__init__(parent=parent)
        self._name_widget = QtGui.QLabel(name)
        self._reward = OneValueWidget(parent=self)
        self._distance = OneValueWidget((0, 255, 0), parent=self)
        self._returns = [ReturnWidget(child_name, discount_factor, parent=self) for child_name in childs_names]
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
    def __init__(self, env, dummy_transition, discount_factor):
        super().__init__()
        # self.setGeometry(50, 50, 1000, 600)
        self._env = env
        self._central_widget = QtGui.QWidget(parent=self)
        self.setCentralWidget(self._central_widget)
        self._goal_0_widget = EnvironmentWidget(*env.renderer._x_lim, *env.renderer._y_lim, self._central_widget)
        self._goal_1_widget = EnvironmentWidget(*env.renderer._x_lim, *env.renderer._y_lim, self._central_widget)
        self._environment_widget = EnvironmentWidget(*env.renderer._x_lim, *env.renderer._y_lim, self._central_widget)
        self._agents = self._create_agents_widget(dummy_transition, discount_factor)
        self._agents[0]._distance.setYRange(-1, 2)
        [a._distance.setYRange(-0.1, 1) for a in self._agents[1:]]
        self._layout = QtGui.QGridLayout(self._central_widget)
        self._layout.addWidget(self._goal_0_widget, 0, 0, 1, 1)
        self._layout.addWidget(self._environment_widget, 1, 0, 1, 1)
        self._layout.addWidget(self._goal_1_widget, 2, 0, 1, 1)
        for i, agent in enumerate(self._agents):
            self._layout.addWidget(agent, 0, i + 1, 4, 1)

    def _create_agents_widget(self, dummy_transition, discount_factor):
        if not len(dummy_transition["childs"]):
            return []
        name = [key for key in dummy_transition if key != "childs"][0]
        childs_names = [[key for key in t if key != "childs"][0] for t in dummy_transition["childs"]]
        return [NonLeafAgentWidget(name, discount_factor, childs_names, parent=self._central_widget)] + \
                sum([self._create_agents_widget(t, discount_factor)
                    for t in dummy_transition["childs"]], [])

    def _refresh_agents(self, transition, agents):
        if not len(transition["childs"]):
            return agents
        childs_names = [[key for key in t if key != "childs"][0] for t in transition["childs"]]
        reward = transition["childs"][0][childs_names[0]]["reward"][0]
        distance = transition["childs"][0][childs_names[0]]["mean_distance_to_goal"]
        predicted_returns = [child[name]["predicted_return"][0] for child, name in zip(transition["childs"], childs_names)]
        targets = [child[name]["critic_target"][0] for child, name in zip(transition["childs"], childs_names)]
        agents[0].refresh_data(distance, reward, predicted_returns, targets)
        agents = agents[1:]
        for t in transition["childs"]:
            agents = self._refresh_agents(t, agents)
        return agents

    def refresh_data(self, transition):
        self._refresh_agents(transition, self._agents)
        self._environment_widget.refresh_data(self._env.ploting_data)

    def refresh_goal(self, ploting_data_0, ploting_data_1):
        self._goal_0_widget.refresh_data(ploting_data_0)
        self._goal_1_widget.refresh_data(ploting_data_1)


class Display:
    def __init__(self, worker, training=False):
        self.worker = worker
        self._count = 0
        self.total_count = 0
        self._sequence_length = worker.sequence_length
        self._training = training
        self.goal = self.worker.get_goal()
        self.state_0 = self.worker.get_state()
        self.gstate_0 = self.worker.get_gstate()
        self.behaviour_fetches = self.worker.display_training_behaviour_fetches if training else self.worker.display_testing_behaviour_fetches
        feed_dict = self.worker.behaviour_feed_dict(self.goal, self.state_0, self.gstate_0)
        # feed_dict feeds root with data from 'state', 'gstate' and 'goal'
        self._transition_0 = self.worker.sess.run(self.worker.training_behaviour_fetches, feed_dict=feed_dict)
        # set action
        action = self.worker.to_action(self._transition_0)
        self.worker.env.set_speeds(action)
        # run environment step
        self.worker.env.env_step()
        self.app = QtGui.QApplication(sys.argv)
        self.window = Window(self.worker.env, self._transition_0, self.worker.discount_factor)
        self.window.resize(1536, 864)
        self.window.show()

    def show(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self)
        timer.start(0)
        self.app.exec_()

    def save(self, path, n_frames=None, length_in_sec=None):
        if not n_frames and not length_in_sec:
            length_in_sec = 240
        if not n_frames:
            n_frames = length_in_sec * 25
        pixmap = QtGui.QPixmap.grabWindow(self.window._central_widget.winId())
        height = pixmap.height()
        width = pixmap.width()
        channels_count = 3
        size = height * width * channels_count
        with get_writer(path, fps=25) as writer:
            def save_func():
                self()
                pixmap = QtGui.QPixmap.grabWindow(self.window._central_widget.winId())
                image = pixmap.toImage()
                image = image.convertToFormat(QtGui.QImage.Format_RGB888)
                bits = image.bits()
                bits.setsize(size)
                arr = np.frombuffer(bits, np.uint8).reshape((height, width, channels_count))
                writer.append_data(arr)
                if self.total_count >= n_frames:
                    self.app.exit()

            timer = QtCore.QTimer()
            timer.timeout.connect(save_func)
            timer.start(0)
            self.app.exec_()

    def _set_count(self, val):
        self._count = val % self._sequence_length

    def _get_count(self):
        return self._count

    count = property(_get_count, _set_count)

    def __call__(self):
        if self.count == 0:
            # sample new goal
            self.worker.randomize_env()
            self.goal, (ploting_data_0, ploting_data_1) = self.worker.goals_buffer.sample()
            self.goal = self.goal[np.newaxis]
            self.window.refresh_goal(ploting_data_0, ploting_data_1)
            # register goals
            self.state_0 = self.worker.get_state()
            self.gstate_0 = self.worker.get_gstate()
            self.behaviour_fetches = self.worker.display_training_behaviour_fetches if self._training else self.worker.display_testing_behaviour_fetches
            feed_dict = self.worker.behaviour_feed_dict(self.goal, self.state_0, self.gstate_0)
            # feed_dict feeds root with data from 'state', 'gstate' and 'goal'
            transition_0 = self.worker.sess.run(self.worker.training_behaviour_fetches, feed_dict=feed_dict)
            # set action
            action = self.worker.to_action(transition_0)
            self.worker.env.set_speeds(action)
            # run environment step
            self.worker.env.env_step()
        # get states
        self.state_1 = self.worker.get_state()
        self.gstate_1 = self.worker.get_gstate()
        feed_dict = self.worker.display_feed_dict(self.goal, self.state_0, self.gstate_0, self.state_1, self.gstate_1)
        # feed_dict feeds root with data from 'state', 'gstate' and 'goal'
        transition, global_step = self.worker.sess.run([self.behaviour_fetches, self.worker.global_step], feed_dict=feed_dict)
        # update window
        self.window.refresh_data(transition)
        # set action
        action = self.worker.to_action(transition)
        self.worker.env.set_speeds(action)
        # run environment step
        self.worker.env.env_step()
        # rotate states
        self.state_0 = self.state_1
        self.gstate_0 = self.gstate_1
        self.worker.goals_buffer.register_one(self.gstate_0, self.worker.env.ploting_data)
        self.count += 1
        self.total_count += 1
