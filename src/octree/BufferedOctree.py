from typing import Optional, List

import numpy as np

from src.octree.BufferedOctreeBranch import BufferedOctreeBranch
from src.octree.Octree import Octree


class BufferedOctree(Octree):
    def __init__(self, resolution: float, buffer_size: int):
        super().__init__(resolution)
        self.buffer_size = buffer_size
        self.selected_buffer = 0
        self.root_node: Optional[BufferedOctreeBranch] = None

    def insert_points(self, points, store_in_leaf=False, selected_buffer=None, *args, **kwargs) -> None:
        selected_buffer = selected_buffer if selected_buffer is not None else self.selected_buffer
        super().insert_points(points, store_in_leaf, selected_buffer=selected_buffer)

    def expend_tree(self, point, selected_buffer=None, *args, **kwargs):
        super().expend_tree(point, selected_buffer=selected_buffer, *args, **kwargs)
        for index, children in enumerate(self.root_node.get_children_list(selected_buffer=selected_buffer)):
            if children is not None:
                for i in range(self.buffer_size):
                    if i == selected_buffer:
                        continue
                    self.root_node.set_children(index, children, selected_buffer=i)

    def create_root_node(self, depth, new_bound):
        return BufferedOctreeBranch(depth, new_bound.origin, new_bound.size, None, self.buffer_size)

    def serialize(self, selected_buffer=None) -> (np.ndarray, int, List[float], float):
        selected_buffer = selected_buffer if selected_buffer is not None else self.selected_buffer
        return super().serialize(selected_buffer=selected_buffer)

    def deserialize(self,
                    bit_pattern_list,
                    resolution,
                    depth,
                    origin,
                    size,
                    selected_buffer=None,
                    *args,
                    **kwargs) -> (np.ndarray, int, List[float], float):
        selected_buffer = selected_buffer if selected_buffer is not None else self.selected_buffer
        super().deserialize(bit_pattern_list, resolution, depth, origin, size, selected_buffer=selected_buffer)
