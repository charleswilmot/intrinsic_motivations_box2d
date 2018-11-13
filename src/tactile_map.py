import numpy as np


def norm(point):
    a, b = point
    return np.sqrt(a * a + b * b)


def norm2(point):
    a, b = point
    return a * a + b * b


def normalize(point):
    return point / norm(point)


def project_on_edge(point, edge):
    (x1, y1), (x2, y2) = edge
    x3, y3 = point
    dx = x2 - x1
    dy = y2 - y1
    d2 = dx * dx + dy * dy
    nx = ((x3 - x1) * dx + (y3 - y1) * dy) / d2
    nx = min(1, max(0, nx))
    return np.array([dx * nx + x1, dy * nx + y1]), nx


def project_on_edges(edges, point):
    edges_coord = [project_on_edge(point, edge) for edge in edges]
    distances2 = [norm2(point - edge_coord[0]) for edge, edge_coord in zip(edges, edges_coord)]
    closest_edge_index = np.argmin(distances2)
    return closest_edge_index, edges_coord[closest_edge_index][1]


def edges_length(edges):
    return np.array([norm(a - b) for a, b in edges])


class TactileSensor:
    def __init__(self, body, n_touch_cells):
        self.body = body
        self.n_touch_cells = n_touch_cells
        vertices = np.array(body.fixtures[0].shape.vertices)
        self.edges = np.array(list(zip(vertices, vertices[list(range(1, len(vertices))) + [0]])))
        self.edges_norm = np.array([norm(e[1] - e[0]) for e in self.edges])
        self._map = np.zeros(n_touch_cells)
        self._map_updated = True
        self._cumulative_perimeter = np.cumsum(edges_length(self.edges))
        self.body_perimeter = self._cumulative_perimeter[-1]
        self.step = self.body_perimeter / self.n_touch_cells
        index_edge_end = [int(np.floor(a / self.step)) for a in self._cumulative_perimeter]
        self._edges_to_slice = [slice(a + 1, b + 1) for a, b in zip([-1] + index_edge_end, index_edge_end)]

    def _assign_float(self, findex, value):
        index = np.floor(findex)
        if index == findex:
            index = int(index) % self.n_touch_cells
            self._map[index] = value
        else:
            ratio = findex - index
            index = int(index)
            index_next = (index + 1) % self.n_touch_cells
            self._map[index] = value * (1 - ratio)
            self._map[index_next] = value * ratio

    def _get_perimeter(self, proj):
        n = self.edges_norm[proj[0]]
        return self._cumulative_perimeter[proj[0] - 1] + proj[1] * n if proj[0] > 0 else proj[1] * n

    def compute_map(self):
        self._map[:] = 0
        for point in self._points:
            proj = project_on_edges(self.edges, point)
            perim = self._get_perimeter(proj)
            self._assign_float(perim / self.step, 1)
        return self._map

    def _get_contact_points(self):
        contacts_world = [ce.contact.worldManifold.points[0]
                          for ce in self.body.contacts if ce.contact.touching]
        contacts_local = [self.body.GetLocalPoint(x)
                          for x in contacts_world]
        return contacts_local

    _points = property(_get_contact_points)

    def compute_edge_map(self, edge_indices):
        tactile_map = self.compute_map()
        return [tactile_map[self._edges_to_slice[edge_index]] for edge_index in edge_indices]

    def edge_map_length(self, edge_index):
        s = self._edges_to_slice[edge_index]
        return s.stop - s.start


class Skin:
    def __init__(self, bodies, order, resolution):
        used_bodies = {k: bodies[k] for k in bodies if k in [a for a, b in order]}
        self.tactile_sensors = {k: TactileSensor(used_bodies[k], resolution) for k in used_bodies}
        self.order = order
        self._emap_lengths = [self.tactile_sensors[i].edge_map_length(j) for i, j in self.order]
        self._emap_cum_lengths = np.cumsum(self._emap_lengths)
        self.length = self._emap_cum_lengths[-1]
        self._map = np.zeros(self.length)

    def compute_map(self):
        self._map[:] = 0
        for a in self.tactile_sensors:
            l = [(i, a, e) for i, (ts, e) in enumerate(self.order) if ts == a]
            maps = self.tactile_sensors[a].compute_edge_map([e for i, ts, e in l])
            for m, (i, ts, e) in zip(maps, l):
                start = 0 if i == 0 else self._emap_cum_lengths[i - 1]
                stop = self._emap_cum_lengths[i]
                self._map[start:stop] = m
        return self._map


if __name__ == "__main__":
    def test_norm():  # point
        p = np.array([2.0, 0.0])
        assert(norm(p) == 2.0)

    def test_normalize():  # point
        p = np.array([2.0, 0.0])
        assert((normalize(p) == np.array([1.0, 0.0])).all())

    def test_project_on_edge():  # point, edge
        p = np.array([2.0, 0.0])
        e = np.array([[0.0, 1.0], [0.0, 0.0]])
        ret = project_on_edge(p, e)[1]
        assert(ret == 1.0)

    def test():
        test_norm()
        test_normalize()
        test_project_on_edge()

    test()
