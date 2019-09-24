import numpy as np


def norm(point):
    a, b = point
    return np.sqrt(a * a + b * b)


def norm2(point):
    a, b = point
    return a * a + b * b


def project_on_edge(point, edge):
    (x1, y1), (x2, y2) = edge
    x3, y3 = point
    dx = x2 - x1
    dy = y2 - y1
    d2 = dx * dx + dy * dy
    nx = ((x3 - x1) * dx + (y3 - y1) * dy) / d2
    nx = min(1, max(0, nx))
    return np.array([dx * nx + x1, dy * nx + y1]), nx


class TactileSensor:
    def __init__(self, body, edge_ids):
        self.body = body
        self.edge_ids = edge_ids
        vertices = np.array(body.fixtures[0].shape.vertices)
        self.edges = np.array(list(zip(vertices, vertices[list(range(1, len(vertices))) + [0]])))[edge_ids]
        edges = np.array(list(zip(vertices, vertices[list(range(1, len(vertices))) + [0]])))
        self.edges = edges[edge_ids]
        self.edges_length = np.array([norm(e[1] - e[0]) for e in self.edges])

    def project_on_edges(self, point):
        edges_coord = [project_on_edge(point, edge) for edge in self.edges]
        distances2 = [norm2(point - edge_coord[0]) for edge, edge_coord in zip(self.edges, edges_coord)]
        closest_edge_index = np.argmin(distances2)
        return closest_edge_index, edges_coord[closest_edge_index][1]

    def get_contacts_per_edge(self):
        ret = []
        for i in self.edge_ids:
            ret.append([])
        for point in self._points:
            which_edge, where_on_edge = self.project_on_edges(point)
            ret[which_edge].append(where_on_edge)
        return [np.array(contacts) * length for contacts, length in zip(ret, self.edges_length)]

    def _get_contact_points(self):
        contacts_world = [ce.contact.worldManifold.points[0]
                          for ce in self.body.contacts if ce.contact.touching]
        contacts_local = [self.body.GetLocalPoint(x)
                          for x in contacts_world]
        return contacts_local

    _points = property(_get_contact_points)


class Skin:
    def __init__(self, bodies, order, resolution):
        self.order = order
        self.resolution = resolution
        used_bodies = {k: bodies[k] for k in bodies if k in [a for a, b in order]}
        bodies_to_edges_ids = {k: [edge_id for body_name, edge_id in order if body_name == k] for k in used_bodies}
        self.tactile_sensors = {k: TactileSensor(used_bodies[k], bodies_to_edges_ids[k]) for k in used_bodies}
        self.reformated_order = []
        for body_name, edge_id in order:
            new_order = 0
            for a, b in self.reformated_order:
                if a == body_name:
                    new_order += 1
            self.reformated_order.append((body_name, new_order))
        edges_length = np.array([0] + [self.tactile_sensors[body_name].edges_length[edge_id] for body_name, edge_id in self.reformated_order])
        self.edges_shift = np.cumsum(edges_length)
        self._map = np.zeros(resolution)

    def compute_map(self):
        all_contacts = {
            body_name: sensor.get_contacts_per_edge()
            for body_name, sensor in self.tactile_sensors.items()
        }
        all_contacts_reordered = [
            all_contacts[body_name][edge_id]
            for body_name, edge_id in self.reformated_order
        ]
        all_contacts_reordered_on_perimeter = [
            positions_on_edges + edge_shift
            for positions_on_edges, edge_shift
            in zip(all_contacts_reordered, self.edges_shift)
        ]
        all_contacts_reordered_on_perimeter = np.concatenate(all_contacts_reordered_on_perimeter, axis=0)
        return self.discretize(all_contacts_reordered_on_perimeter)

    def discretize(self, contacts_on_perimeter):
        self._map[:] = 0
        for contact in contacts_on_perimeter:
            findex = contact * (self.resolution - 1) / self.edges_shift[-1]
            index = int(np.floor(findex))
            val = findex - index
            self._map[index] += val
            if index + 1 < self.resolution:
                self._map[index + 1] += 1 - val
        return self._map
