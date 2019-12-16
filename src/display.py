import sys
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np


class EnvironmentScene(QtGui.QGraphicsScene):
    def __init__(self, env, parent=None):
        super().__init__(parent=parent)
        self._env = env
        self._polygons = []
        self.refresh_data()

    def refresh_data(self):
        self.clear()
        for key, body in self._env.bodies.items():
            touching = [ce.contact.touching for ce in body.contacts if ce.contact.touching]
            vercs = np.vstack(body.fixtures[0].shape.vertices)
            data = [body.GetWorldPoint(x) for x in vercs]
            polygon = QtGui.QPolygonF()
            for p in data:
                polygon.append(QtCore.QPointF(*p))
            color = (255, 0, 0) if len(touching) > 0 and key in self._env.renderer.tactile_bodies_names else (0, 0, 255)
            self.addPolygon(polygon)
            self._polygons.append(polygon)


class EnvironmentWidget(QtGui.QGraphicsView):
    def __init__(self, env, parent=None):
        super().__init__(parent=parent)
        self._env = env
        self._scene = EnvironmentScene(env, parent=self)
        self.setScene(self._scene)
        self._x = self._env.renderer._x_lim[0]
        self._y = self._env.renderer._y_lim[0]
        self._w = self._env.renderer._x_lim[1] - self._env.renderer._x_lim[0]
        self._h = self._env.renderer._y_lim[1] - self._env.renderer._y_lim[0]
        self.fitInView()

    def fitInView(self):
        super().fitInView(self._x, self._y, self._w, self._h, 1)

    def resizeEvent(self, e):
        self.fitInView()

    def refresh_data(self):
        self._scene.refresh_data()


class ReadoutWidget(pg.GraphicsLayoutWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._view = self.addViewBox()
        self._scatter = pg.ScatterPlotItem(x=np.arange(10), y=np.random.uniform(size=10))
        self._view.addItem(self._scatter)
        self._x = None

    def refresh_data(self, new):
        if self._x is None:
            self._x = np.arange(len(new))
        self._scatter.setData(self._x, new) ## have a look at http://www.pyqtgraph.org/documentation/_modules/pyqtgraph/graphicsItems/ScatterPlotItem.html#ScatterPlotItem.addPoints


class ThreeReadoutsWidget(QtGui.QWidget):
    def __init__(self, parent=None, horizontal=False):
        super().__init__(parent=parent)
        # self.resize(600, 100)
        self._state = ReadoutWidget(parent=self)
        self._goal = ReadoutWidget(parent=self)
        self._gstate = ReadoutWidget(parent=self)
        self._layout = QtGui.QHBoxLayout(self) if horizontal else QtGui.QVBoxLayout(self)
        self._layout.addWidget(self._state)
        self._layout.addWidget(self._goal)
        self._layout.addWidget(self._gstate)
        self.setLayout(self._layout)
        self._layout.setSpacing(0)
        self._layout.setMargin(0)

    def refresh_data(self, state, goal, gstate):
        self._state.refresh_data(state)
        self._goal.refresh_data(goal)
        self._gstate.refresh_data(gstate)


class RewardWidget(pg.PlotWidget):
    def __init__(self, lookback=15, parent=None):
        super().__init__(parent=parent)
        self._x = np.arange(-lookback + 1, 1)
        self._reward_buffer = np.zeros(lookback)
        self.setWindowTitle('reward')
        self.setRange(QtCore.QRectF(-lookback, -1, lookback, 2))
        self.setLabel('bottom', 'iterations')
        self._reward_curve = self.plot(pen=pg.mkPen(255, 128, 128))

    def refresh_data(self, reward):
        self._reward_buffer[:-1] = self._reward_buffer[1:]
        self._reward_buffer[-1] = reward
        self._reward_curve.setData(x=self._x, y=self._reward_buffer)



class ReturnWidget(pg.PlotWidget):
    def __init__(self, discount_factor, lookback=15, parent=None):
        super().__init__(parent=parent)
        self._x = np.arange(-lookback + 1, 1)
        self._return_buffer = np.zeros(lookback)
        self._prediction_buffer = np.zeros(lookback)
        self._prediction_part = np.zeros(lookback)
        self._gammas = np.cumprod(np.full(lookback, discount_factor))[::-1] / discount_factor
        self._discount_factor = discount_factor
        self.setWindowTitle('return')
        self.setRange(QtCore.QRectF(-lookback, -1, lookback, 2))
        self.setLabel('bottom', 'iterations')
        self._return_curve = self.plot(pen=pg.mkPen(255, 255, 0))
        self._prediction_curve = self.plot(pen=pg.mkPen(51, 153, 255))

    def refresh_data(self, reward, prediction):
        self._prediction_buffer[:-1] = self._prediction_buffer[1:]
        self._prediction_buffer[-1] = prediction
        self._return_buffer += self._gammas * reward - self._prediction_part
        self._return_buffer[:-1] = self._return_buffer[1:]
        self._prediction_part = prediction * self._gammas
        self._return_buffer += self._prediction_part
        self._return_buffer[-1] = prediction
        self._return_curve.setData(x=self._x, y=self._return_buffer)
        self._prediction_curve.setData(x=self._x, y=self._prediction_buffer)


class RewardReturnWidget(QtGui.QWidget):
    def __init__(self, discount_factor, parent=None):
        super().__init__(parent=parent)
        self._reward = RewardWidget(parent=self)
        self._return = ReturnWidget(discount_factor, parent=self)
        self._layout = QtGui.QHBoxLayout(self)
        self._layout.addWidget(self._reward)
        self._layout.addWidget(self._return)
        self._layout.setSpacing(0)
        self._layout.setMargin(0)
        self.setLayout(self._layout)

    def refresh_data(self, reward, predicted_return):
        self._reward.refresh_data(reward)
        self._return.refresh_data(reward, predicted_return)


class NonLeafAgentWidget(QtGui.QWidget):
    def __init__(self, discount_factor, parent=None):
        super().__init__(parent=parent)
        N = 2
        self._rwidget = RewardReturnWidget(discount_factor, parent=self)
        self._readouts = [ThreeReadoutsWidget(parent=self) for i in range(N)]
        self._layout = QtGui.QGridLayout(self)
        self._layout.addWidget(self._rwidget, 0, 0, 1, N)
        for i, readout in enumerate(self._readouts):
            self._layout.addWidget(readout, 1, i, 3, 1)
        self._layout.setSpacing(0)
        self._layout.setMargin(0)
        self.setLayout(self._layout)

    def refresh_data(self, reward, predicted_return, states, goals, gstates):
        self._rwidget.refresh_data(reward, predicted_return)
        for readout, state, goal, gstate in zip(self._readouts, states, goals, gstates):
            readout.refresh_data(state, goal, gstate)


class Window(QtGui.QMainWindow):
    def __init__(self, env, dummy_transition, discount_factor):
        super().__init__()
        # self.setGeometry(50, 50, 1000, 600)
        self._central_widget = QtGui.QWidget(parent=self)
        self.setCentralWidget(self._central_widget)
        self._environment_widget = EnvironmentWidget(env, self._central_widget)
        self._state_goal_gstate = ThreeReadoutsWidget(self._central_widget)
        self._agents = self._create_agents_widget(dummy_transition, discount_factor)
        self._layout = QtGui.QGridLayout(self._central_widget)
        self._layout.addWidget(self._environment_widget, 0, 0, 1, 1)
        self._layout.addWidget(self._state_goal_gstate, 1, 0, -1, 1)
        for i, agent in enumerate(self._agents):
            self._layout.addWidget(agent, 0, i + 1, 4, 1)
        self.show()

    def _create_agents_widget(self, dummy_transition, discount_factor):
        if not len(dummy_transition["childs"]):
            return []
        return [NonLeafAgentWidget(discount_factor, parent=self._central_widget)] + \
                sum([self._create_agents_widget(t, discount_factor)
                    for t in dummy_transition["childs"]], [])

    def _refresh_agents(self, transition, agents):
        if not len(transition["childs"]):
            return agents
        childs_names = sum([[name for name in child if name != "childs"] for child in transition["childs"]], [])
        one_child = transition["childs"][0]
        name = [name for name in one_child if name != "childs"][0]
        one_child = one_child[name]
        reward = one_child["reward"][0]
        predicted_return = one_child["predicted_return"][0]
        states = [child[name]["readout_state"][0] for name, child in zip(childs_names, transition["childs"])]
        goals = [child[name]["readout_goal"][0] for name, child in zip(childs_names, transition["childs"])]
        gstates = [child[name]["readout_gstate"][0] for name, child in zip(childs_names, transition["childs"])]
        agents[0].refresh_data(reward, predicted_return, states, goals, gstates)
        agents = agents[1:]
        for t in transition["childs"]:
            agents = self._refresh_agents(t, agents)
        return agents

    def refresh_data(self, transition):
        self._refresh_agents(transition, self._agents)
        state = transition["all_body"]["state_0"][0]
        goal = transition["all_body"]["goal_0"][0]
        gstate = transition["all_body"]["gstate_0"][0]
        self._state_goal_gstate.refresh_data(state, goal, gstate)
        self._environment_widget.refresh_data()


class Display:
    def __init__(self, worker, training=False):
        self.worker = worker
        self._count = 1
        self._sequence_length = worker.sequence_length
        self.goal = self.worker.get_goal()
        self.state_0 = self.worker.get_state()
        self.gstate_0 = self.worker.get_gstate()
        self.behaviour_fetches = self.worker.display_training_behaviour_fetches if training else self.worker.display_testing_behaviour_fetches
        self.to_goal_buffer = []
        feed_dict = self.worker.behaviour_feed_dict(self.goal, self.state_0, self.gstate_0)
        # feed_dict feeds root with data from 'state', 'gstate' and 'goal'
        transition_0 = self.worker.sess.run(self.worker.training_behaviour_fetches, feed_dict=feed_dict)
        # set action
        action = self.worker.to_action(transition_0)
        self.worker.env.set_speeds(action)
        # run environment step
        self.worker.env.env_step()
        app = QtGui.QApplication(sys.argv)
        self.window = Window(worker.env, transition_0, worker.discount_factor)
        timer = QtCore.QTimer()
        timer.timeout.connect(self)
        timer.start(0)
        app.exec_()


    def _set_count(self, val):
        self._count = val % self._sequence_length

    def _get_count(self):
        return self._count

    count = property(_get_count, _set_count)

    def __call__(self):
        if self.count == 0:
            # sample new goal
            self.worker.randomize_env()
            self.goal = self.worker.get_goal()
            # register goals
            self.worker.goals_buffer.register(self.to_goal_buffer)
            self.to_goal_buffer.clear()
        # get states
        self.state_1 = self.worker.get_state()
        self.gstate_1 = self.worker.get_gstate()
        self.to_goal_buffer.append(self.gstate_0[0])
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
        self.count += 1
