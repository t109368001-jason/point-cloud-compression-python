import logging
from typing import Optional, List

import numpy as np

from src.octree.BufferedOctreeBranch import BufferedOctreeBranch
from src.octree.Octree import Octree


class BufferedOctree(Octree):
    def __init__(self, resolution: float, buffer_size: int):
        super().__init__(resolution)
        self.buffer_size = buffer_size
        self.selector = 0
        self.root_node: Optional[BufferedOctreeBranch] = None

    def insert_points(self, points, store_in_leaf=False, selector=None, *args, **kwargs) -> None:
        selector = selector if selector is not None else self.selector
        super().insert_points(points, store_in_leaf, selector=selector)

    def expend_tree(self, point, selector=None, *args, **kwargs):
        super().expend_tree(point, selector=selector, *args, **kwargs)
        for index, children in enumerate(self.root_node.get_children_list(selector=selector)):
            if children is not None:
                for i in range(self.buffer_size):
                    if i == selector:
                        continue
                    self.root_node.set_children(index, children, selector=i)

    def create_root_node(self, depth, new_bound):
        return BufferedOctreeBranch(depth, new_bound.origin, new_bound.size, None, self.buffer_size)

    def serialize(self, selector=None) -> (np.ndarray, int, List[float], float):
        selector = selector if selector is not None else self.selector
        result = super().serialize(selector=selector)
        return result

    def deserialize(self,
                    bit_pattern_list,
                    resolution,
                    depth,
                    origin,
                    size,
                    selector=None,
                    *args,
                    **kwargs) -> (np.ndarray, int, List[float], float):
        selector = selector if selector is not None else self.selector
        super().deserialize(bit_pattern_list, resolution, depth, origin, size, selector=selector)

    def motion_estimation(self, depth, selector_i, selector_f):
        logging.info("start")
        bit_pattern_list = self.root_node.serialize(depth=depth + 1, selector=selector_f)
        indices = self.root_node.motion_estimation(depth, selector_i, selector_f)
        logging.info("end")
        return bit_pattern_list, indices

    def motion_compensation(self, depth, bit_pattern_list, indices, selector_i, selector_p):
        logging.info("start")
        self.root_node.deserialize(depth=depth + 1, bit_pattern_list=bit_pattern_list, selector=selector_p)
        self.root_node.motion_compensation(depth=depth, indices=indices, selector_i=selector_i, selector_p=selector_p)
        logging.info("end")
