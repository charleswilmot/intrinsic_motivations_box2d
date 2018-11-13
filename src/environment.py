import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import jsonpybox2d as json2d
import numpy as np
import tactile_map as tm
import pid


class Environment(object):
    """ 2D physics using box2d and a json conf file
    """
    def __init__(self, world_file, skin_order, skin_resolution, xlim, ylim, dpi, dt=1 / 120.0):
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
        self.world = world
        self.bodies = bodies
        self.joints = joints
        self.joint_pids = {key: pid.PID(dt=self.dt)
                           for key in self.joints}
        self._joint_keys = [key for key in self.joints]
        self._proprio_buf = np.zeros((len(self.joints), 2))
        self.skin = tm.Skin(self.bodies, skin_order, skin_resolution)
        self._joints_in_position_mode = set()
        tactile_bodies_names = set([body_name for body_name, edge in skin_order])
        self.renderer = Renderer(self.bodies, xlim, ylim, tactile_bodies_names=tactile_bodies_names, dpi=dpi)

    def set_speeds(self, speeds):
        for key in speeds:
            if key in self.joints:
                if key in self._joints_in_position_mode:
                    self._joints_in_position_mode.remove(key)
                self.joints[key].motorSpeed = speeds[key]

    def set_positions(self, positions):
        for key in positions:
            if key in self.joint_pids:
                self._joints_in_position_mode.add(key)
                current = self.joints[key].angle
                position = positions[key]
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
                self.joint_pids[key].setpoint = pos

    def step(self):
        for key in self._joints_in_position_mode:
            self.joint_pids[key].step(self.joints[key].angle)
            self.joints[key].motorSpeed = self.joint_pids[key].output
        self.world.Step(self.dt, self._vel_iters, self._pos_iters)

    def _get_state_vision(self):
        return self.renderer.step()

    def _get_state_tactile(self):
        return self.skin.compute_map()

    def _get_state_proprio(self):
        for i, key in enumerate(self._joint_keys):
            self._proprio_buf[i] = \
                self.joints[key].motorSpeed,\
                self.joints[key].angle
        self._proprio_buf[:, 1] %= 2 * np.pi
        self._proprio_buf[:, 1] -= np.pi
        return self._proprio_buf

    def _get_state(self):
        vision = self._get_state_vision()
        tactile_map = self._get_state_tactile()
        proprioception = self._get_state_proprio()
        return vision, proprioception, tactile_map

    state = property(_get_state)
    proprioception = property(_get_state_proprio)
    vision = property(_get_state_vision)
    tactile = property(_get_state_tactile)


class Renderer:
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


if __name__ == "__main__":
    import viewer

    win = viewer.Window()

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

    for i in range(100):
        actions = {
            "Arm1_to_Arm2_Left": np.random.uniform(-1, 1),
            "Ground_to_Arm1_Left": np.random.uniform(-1, 1),
            "Arm1_to_Arm2_Right": np.random.uniform(-1, 1),
            "Ground_to_Arm1_Right": np.random.uniform(-1, 1)
        }
        env.set_positions(actions)
        for j in range(300):
            if j % 10 == 0:
                win.update(*env.state)
            env.step()
