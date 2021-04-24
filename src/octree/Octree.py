from typing import Optional, List

import numpy as np
from tqdm import tqdm

from src.octree.OctreeBranch import OctreeBranch
from src.octree.OctreeLeaf import OctreeLeaf


class Octree:
    def __init__(self, resolution: float):
        self.resolution = resolution
        self.root_node: Optional[OctreeBranch] = None

    def insert_points(self, points, store_in_leaf=False) -> None:
        for point in tqdm(points, desc="Inserting"):
            if self.root_node is None:
                new_leaf = OctreeLeaf(0, point, self.resolution, None)
                new_bound = new_leaf.get_expend_bound(point)
                self.root_node = OctreeBranch(1, new_bound.origin,
                                              new_bound.size, None)
                new_index = new_bound.get_index(point)
                new_leaf.parent = self.root_node
                self.root_node.set_children(new_index, new_leaf)
            elif not self.root_node.in_bound(point):
                new_bound = self.root_node.get_expend_bound(point)
                new_root_node = OctreeBranch(self.root_node.depth + 1,
                                             new_bound.origin, new_bound.size,
                                             None)
                new_index = new_bound.get_index(self.root_node.origin)
                new_root_node.set_children(new_index, self.root_node)
                self.root_node.parent = new_root_node
                self.root_node = new_root_node
            self.root_node.insert_point(point, store_in_leaf)

    def serialize(self) -> (np.ndarray, int, List[float], float):
        bit_pattern_list = self.root_node.serialize()
        return bit_pattern_list, self.resolution, self.root_node.depth, self.root_node.origin, self.root_node.size

    def deserialize(self, bit_pattern_list, resolution, depth, origin, size) -> (np.ndarray, int, List[float], float):
        self.resolution = resolution
        self.root_node = OctreeBranch(depth, origin, size, None)
        self.root_node.deserialize(bit_pattern_list)
