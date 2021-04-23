import numpy as np


class Bound:
    def __init__(self, origin: list = None, size: float = None):
        self.origin = origin if isinstance(origin,
                                           list) else [x for x in origin[0:3]]
        self.size = size

    def in_bound(self, point: np.ndarray) -> bool:
        if self.origin is None:
            raise ValueError("center is None")
        if self.size is None or self.size <= 0:
            raise ValueError("size is None or size <= 0")
        if point[0] >= self.get_max_x():
            return False
        if point[0] < self.get_min_x():
            return False
        if point[1] >= self.get_max_y():
            return False
        if point[1] < self.get_min_y():
            return False
        if point[2] >= self.get_max_z():
            return False
        if point[2] < self.get_min_z():
            return False
        return True

    def get_expend_bound(self, point):
        bound = Bound(self.origin.copy(), self.size * 2)
        if point[0] < self.origin[0]:
            bound.origin[0] -= self.size
        if point[1] < self.origin[1]:
            bound.origin[1] -= self.size
        if point[2] < self.origin[2]:
            bound.origin[2] -= self.size
        return bound

    def get_sub_bound(self, index):
        bound = Bound(self.origin.copy(), self.size / 2)
        if index % 2 == 1:
            bound.origin[0] += bound.size
        if int(index / 2) % 2 == 1:
            bound.origin[1] += bound.size
        if int(index / 4) % 2 == 1:
            bound.origin[2] += bound.size
        return bound

    def get_index(self, point) -> int:
        index = 0
        if point[0] >= (self.origin[0] + self.size / 2):
            index += 1
        if point[1] >= (self.origin[1] + self.size / 2):
            index += 2
        if point[2] >= (self.origin[2] + self.size / 2):
            index += 4
        return index

    def get_max_x(self):
        return self.origin[0] + self.size

    def get_min_x(self):
        return self.origin[0]

    def get_max_y(self):
        return self.origin[1] + self.size

    def get_min_y(self):
        return self.origin[1]

    def get_max_z(self):
        return self.origin[2] + self.size

    def get_min_z(self):
        return self.origin[2]
