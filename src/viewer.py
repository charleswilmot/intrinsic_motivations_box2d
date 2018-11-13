import matplotlib.pyplot as plt
import numpy as np
import time


plt.ion()


class Viewer:
    def __init__(self, path, fps=25):
        self.path = path + "/"
        self.fps = fps
        self.data = database.Database(self.path)
        self.window = Window()

    def __call__(self):
        t0 = time.time()
        i = 0
        while i < len(self.data):
            self.display(i)
            t1 = time.time()
            delta = t1 - t0
            i = int(self.fps * delta)

    def display(self, i):
        vision, proprioception, tactile_map = self.data[i]
        self.window.update(vision, proprioception, tactile_map)


class Window:
    def __init__(self):
        self.fig = plt.figure()
        self.ax_vision = self.fig.add_subplot(211)
        self.ax_tactile = self.fig.add_subplot(223)
        self.ax_joints = self.fig.add_subplot(224)
        self.ax_joints.set_ylim([-4, 4])
        self._axes_initialized = False
        self._vision_lim = [[0, 255]]
        self._tactile_lim = [[-0.1, 1.1]]
        self._proprioception_lim = [[-4, 4]]

    def set_vision_lim(self, *lim):
        self._vision_lim = lim
        if self._axes_initialized:
            self.ax_vision_image.set_clim(*lim)

    def set_tactile_lim(self, *lim):
        self._tactile_lim = lim
        if self._axes_initialized:
            self.ax_tactile.set_ylim(*lim)

    def set_proprioception_lim(self, *lim):
        self._proprioception_lim = lim
        if self._axes_initialized:
            self.ax_joints.set_ylim(*lim)

    def __initaxes__(self, vision, proprioception, tactile_map):
        tactile_map[0] = 1
        self.ax_tactile_line, = self.ax_tactile.plot(tactile_map)
        self.ax_vision_image = self.ax_vision.imshow(vision)
        self.ax_joints_speed, self.ax_joints_angle = self.ax_joints.plot(proprioception, "o")
        self.ax_vision_image.set_clim(*self._vision_lim)
        self.ax_vision_image.axes.get_xaxis().set_visible(False)
        self.ax_vision_image.axes.get_yaxis().set_visible(False)
        self.ax_tactile.set_ylim(*self._tactile_lim)
        self.ax_tactile.axes.get_yaxis().set_ticks([0, 1])
        self.ax_tactile.set_title("Skin sensor")
        self.ax_joints.set_ylim(*self._proprioception_lim)
        self.ax_joints.axes.get_xaxis().set_ticks(range(proprioception.shape[0]))
        self.ax_joints.set_title("Joints speed/position")
        self.fig.show()
        self._axes_initialized = True

    def update(self, vision, proprioception, tactile_map):
        if self._axes_initialized:
            self.ax_vision_image.set_data(vision)
            self.ax_tactile_line.set_ydata(tactile_map)
            self.ax_joints_speed.set_ydata(proprioception[:, 0])
            self.ax_joints_angle.set_ydata(proprioception[:, 1])
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        else:
            self.__initaxes__(vision, proprioception, tactile_map)


if __name__ == "__main__":
    v = Viewer("./data/2_arms_HD_recording/", fps=60)
    v()
