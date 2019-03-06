import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import jsonpybox2d as json2d
import numpy as np
import tactile_map as tm
import pid
from PIL import Image, ImageDraw


def discretize(arr, mini, maxi, n):
    discrete = np.tile(np.linspace(mini, maxi, n), list(arr.shape) + [1])
    discrete -= np.expand_dims(arr, -1)
    discrete = np.cos(np.pi * discrete / (maxi - mini) - np.pi * (maxi + mini) / (maxi - mini)) ** 20
    return discrete


class Environment(object):
    """ 2D physics using box2d and a json conf file
    """
    def __init__(self, world_file, skin_order, skin_resolution, xlim, ylim, dpi, env_step_length,
                 dt=1 / 120.0, n_discrete=32):
        """

            :param world_file: the json file from which all objects are created
            :type world_file: string

            :param dt: the amount of time to simulate, this should not vary.
            :type dt: float

            :param pos_iters: for the velocity constraint solver.
            :type pos_iters: int

            :param vel_iters: for the position constraint solver.
            :type vel_iters: int

        """
        world, bodies, joints = json2d.createWorldFromJson(world_file)
        self.dt = dt
        self._vel_iters = 6
        self._pos_iters = 2
        self._dpi = dpi
        self.env_step_length = env_step_length
        self._n_discrete = n_discrete
        self.world = world
        self.bodies = bodies
        self.joints = joints
        self.joint_pids = {key: pid.PID(dt=self.dt)
                           for key in self.joints}
        self._joint_keys = [key for key in sorted(self.joints)]
        self._joint_keys.sort()
        self._buf_positions = np.zeros(len(self.joints))
        self._buf_target_positions = np.zeros(len(self.joints))
        self._buf_speeds = np.zeros(len(self.joints))
        self.skin = tm.Skin(self.bodies, skin_order, skin_resolution)
        self._joints_in_position_mode = set()
        tactile_bodies_names = set([body_name for body_name, edge in skin_order])
        self.renderer = Renderer(self.bodies, xlim, ylim, tactile_bodies_names=tactile_bodies_names, dpi=dpi)
        self._computed_vision = False
        self._computed_tactile = False
        self._computed_positions = False
        self._computed_discrete_positions = False
        self._computed_speeds = False
        self._computed_discrete_speeds = False
        self._computed_target_positions = False
        self._computed_discrete_target_positions = False

    def set_speeds(self, speeds):
        for key in speeds:
            if key in self.joints:
                if key in self._joints_in_position_mode:
                    self._joints_in_position_mode.remove(key)
                self.joints[key].motorSpeed = np.float64(speeds[key])

    def set_positions_old(self, positions):
        for i, key in enumerate(self._joint_keys):
            if key in positions:
                self._joints_in_position_mode.add(key)
                current = self.joints[key].angle
                position = positions[key]
                if self.joints[key].limitEnabled:
                    pos = position
                else:
                    abs_position = (position % (2 * np.pi)) - np.pi
                    abs_current = (current % (2 * np.pi)) - np.pi
                    diff = abs_current - abs_position
                    inner = abs(diff)
                    outer = 2 * np.pi - inner
                    if inner > outer:
                        if diff > 0:
                            delta = 2 * np.pi - inner
                        else:
                            delta = -2 * np.pi + inner
                    else:
                        if diff > 0:
                            delta = -inner
                        else:
                            delta = inner
                    pos = current + delta
                self._buf_target_positions[i] = pos
                self.joint_pids[key].setpoint = pos

    def set_positions(self, positions):
        threshold = 0.05
        for i, key in enumerate(self._joint_keys):
            if key in positions:
                self._joints_in_position_mode.add(key)
                pos = positions[key]
                if self.joints[key].limitEnabled:
                    min_lim, max_lim = self.joints[key].limits
                    if pos < min_lim + threshold:
                        pos = min_lim - threshold * threshold / (pos - 2 * threshold - min_lim)
                    elif pos > max_lim - threshold:
                        pos = max_lim - threshold * threshold / (pos + 2 * threshold - max_lim)
                self._buf_target_positions[i] = pos
                self.joint_pids[key].setpoint = pos
        self._computed_discrete_target_positions = False

    def step(self):
        for key in self._joints_in_position_mode:
            self.joint_pids[key].step(self.joints[key].angle)
            self.joints[key].motorSpeed = np.clip(self.joint_pids[key].output, -np.pi, np.pi)
        self.world.Step(self.dt, self._vel_iters, self._pos_iters)
        self._computed_vision = False
        self._computed_tactile = False
        self._computed_positions = False
        self._computed_discrete_positions = False
        self._computed_speeds = False
        self._computed_discrete_speeds = False
        self._computed_target_positions = False
        self._computed_discrete_target_positions = False

    def env_step(self):
        for i in range(self.env_step_length):
            self.step()

    def _get_state_vision(self):
        if self._computed_vision:
            return self._buf_vision
        else:
            self._buf_vision = self.renderer.step()
            self._computed_vision = True
            return self._buf_vision

    def _get_state_tactile(self):
        if self._computed_tactile:
            return self._buf_tactile
        else:
            self._buf_tactile = self.skin.compute_map()
            self._computed_tactile = True
            return self._buf_tactile

    def _get_state_positions_old(self):
        if self._computed_positions:
            return self._buf_positions
        else:
            for i, key in enumerate(self._joint_keys):
                self._buf_positions[i] = self.joints[key].angle
            self._buf_positions %= 2 * np.pi
            self._buf_positions -= np.pi
            self._computed_positions = True
            return self._buf_positions

    def _get_state_positions(self):
        if self._computed_positions:
            return self._buf_positions
        else:
            for i, key in enumerate(self._joint_keys):
                self._buf_positions[i] = self.joints[key].angle
            self._computed_positions = True
            return self._buf_positions

    def _get_state_discrete_positions(self):
        if self._computed_discrete_positions:
            return self._buf_discrete_positions
        else:
            self._buf_discrete_positions = discretize(self.positions, -np.pi, np.pi, self._n_discrete)
            self._computed_discrete_positions = True
            return self._buf_discrete_positions

    def _get_state_speeds(self):
        if self._computed_speeds:
            return self._buf_speeds
        else:
            for i, key in enumerate(self._joint_keys):
                self._buf_speeds[i] = self.joints[key].speed
            self._computed_speeds = True
            return self._buf_speeds

    def _get_state_discrete_speeds(self):
        if self._computed_discrete_speeds:
            return self._buf_discrete_speeds
        else:
            self._buf_discrete_speeds = discretize(self.speeds, -np.pi, np.pi, self._n_discrete)
            self._computed_discrete_speeds = True
            return self._buf_discrete_speeds

    def _get_state_target_positions(self):
        return self._buf_target_positions

    def _get_state_discrete_target_positions(self):
        if self._computed_discrete_target_positions:
            return self._buf_discrete_target_positions
        else:
            self._buf_discrete_target_positions = discretize(self.target_positions, -np.pi, np.pi, self._n_discrete)
            self._computed_discrete_target_positions = True
            return self._buf_discrete_target_positions

    def _get_state(self):
        vision = self.vision
        positions = self.positions
        speeds = self.speeds
        tactile_map = self.tactile
        return vision, positions, speeds, tactile_map

    state = property(_get_state)
    positions = property(_get_state_positions)
    discrete_positions = property(_get_state_discrete_positions)
    speeds = property(_get_state_speeds)
    discrete_speeds = property(_get_state_discrete_speeds)
    target_positions = property(_get_state_target_positions)
    discrete_target_positions = property(_get_state_discrete_target_positions)
    vision = property(_get_state_vision)
    tactile = property(_get_state_tactile)


class RendererOld:
    def __init__(self, bodies, xlim, ylim, dpi, tactile_bodies_names=[]):
        self.bodies = bodies
        self.tactile_bodies_names = tactile_bodies_names
        self.fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]), dpi=dpi)
        self.fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        self.ax = self.fig.add_subplot(111, aspect="equal")
        self.ax.axis('off')
        self.polygons = {}
        for key in self.bodies:
            self.polygons[key] = Polygon([[0, 0]], True)
            self.ax.add_artist(self.polygons[key])
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

    def step(self):
        for key in self.polygons:
            body = self.bodies[key]
            touching = [ce.contact.touching for ce in body.contacts if ce.contact.touching]
            vercs = np.vstack(body.fixtures[0].shape.vertices)
            data = np.vstack([body.GetWorldPoint(vercs[x])
                              for x in range(len(vercs))])
            self.polygons[key].set_xy(data)
            self.polygons[key].set_color(
                (1, 0, 0) if len(touching) > 0 and key in self.tactile_bodies_names else
                (0, 0, 1))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # access matplotlib's buffer here and return
        w, h = self.fig.canvas.get_width_height()
        buf = np.fromstring(self.fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (h, w, 4)
        buf = buf[:, :, 1:]
        return buf


class Renderer:
    def __init__(self, bodies, xlim, ylim, dpi, tactile_bodies_names=[]):
        self.bodies = bodies
        self.tactile_bodies_names = tactile_bodies_names
        self._x_lim = xlim
        self._y_lim = ylim
        self._max_x = int(dpi * (xlim[1] - xlim[0]))
        self._max_y = int(dpi * (ylim[1] - ylim[0]))
        self.shape = [self._max_x, self._max_y]
        self.dpi = dpi
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = Image.new('RGB', self.shape, (255, 255, 255))
        self.draw = ImageDraw.Draw(self.buffer)

    def point_to_pix(self, point):
        x, y = point
        X = self._max_x * (x - self._x_lim[0]) / (self._x_lim[1] - self._x_lim[0])
        Y = self._max_y * (y - self._y_lim[0]) / (self._y_lim[1] - self._y_lim[0])
        return X, Y

    def step(self):
        self.reset_buffer()
        for key in self.bodies:
            body = self.bodies[key]
            touching = [ce.contact.touching for ce in body.contacts if ce.contact.touching]
            vercs = np.vstack(body.fixtures[0].shape.vertices)
            data = [self.point_to_pix(body.GetWorldPoint(x)) for x in vercs]
            color = (255, 0, 0) if len(touching) > 0 and key in self.tactile_bodies_names else (0, 0, 255)
            self.draw.polygon(data, fill=color)
        return np.asarray(self.buffer)


if __name__ == "__main__":
    import viewer

    win = viewer.VisionJointsSkinWindow()

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
    env = Environment("../models/two_arms.json", skin_order, skin_resolution, xlim, ylim, dpi=10, dt=1 / 150.0)

    for i in range(1000):
        actions = {
            "Arm1_to_Arm2_Left": np.random.uniform(-2.3, 2.3),
            "Ground_to_Arm1_Left": np.random.uniform(-3.14, 3.14),
            "Arm1_to_Arm2_Right": np.random.uniform(-2.3, 2.3),
            "Ground_to_Arm1_Right": np.random.uniform(-3.14, 3.14)
        }
        env.set_positions(actions)
        for j in range(1000):
            if j % 100 == 0:
                win.update(*env.state)
            env.step()
        # for key in env.joint_pids:
        #     speed1 = env.joint_pids[key].output
        #     speed2 = env.joints[key].speed
        #     if np.abs(speed1) > 0.1:
        #         print("PID", key, speed1)
        #     if np.abs(speed2) > 0.001:
        #         print("ENV", key, speed2)
