import logging
import time
from typing import Optional, List

import numpy as np
from tqdm import tqdm

from src.octree.OctreeBranch import OctreeBranch
from src.octree.OctreeLeaf import OctreeLeaf


class Octree:
    def __init__(self, resolution: float):
        self.resolution = resolution
        self.root_node: Optional[OctreeBranch] = None

    def insert_points(self, points, store_in_leaf=False, *args, **kwargs) -> None:
        logging.info("start")
        time.sleep(0.1)
        for point in tqdm(points, desc="Inserting"):
            if self.root_node is None:
                self.init_root_node(point, *args, **kwargs)
            while not self.root_node.in_bound(point):
                self.expend_tree(point, *args, **kwargs)
            self.root_node.insert_point(point, store_in_leaf, *args, **kwargs)
        logging.info("end")

    def expend_tree(self, point, *args, **kwargs):
        new_bound = self.root_node.get_expend_bound(point)
        new_root_node = self.create_root_node(self.root_node.depth + 1, new_bound)
        new_index = new_bound.get_index(self.root_node.origin)
        new_root_node.set_children(new_index, self.root_node, *args, **kwargs)
        self.root_node.parent = new_root_node
        self.root_node = new_root_node

    def init_root_node(self, point, *args, **kwargs):
        new_leaf = OctreeLeaf(0, point, self.resolution, None)
        new_bound = new_leaf.get_expend_bound(point)
        self.root_node = OctreeBranch(1, new_bound.origin, new_bound.size, None)
        self.root_node = self.create_root_node(1, new_bound)
        new_index = new_bound.get_index(point)
        new_leaf.parent = self.root_node
        self.root_node.set_children(new_index, new_leaf, *args, **kwargs)

    def create_root_node(self, depth, new_bound):
        return OctreeBranch(depth, new_bound.origin, new_bound.size, None)

    def serialize(self, *args, **kwargs) -> (np.ndarray, int, List[float], float):
        bit_pattern_list = self.root_node.serialize(*args, **kwargs)
        return bit_pattern_list, self.resolution, self.root_node.depth, self.root_node.origin, self.root_node.size

    def deserialize(self, bit_pattern_list, resolution, depth, origin, size, *args,
                    **kwargs) -> (np.ndarray, int, List[float], float):
        self.resolution = resolution
        self.root_node = OctreeBranch(depth, origin, size, None)
        self.root_node.deserialize(bit_pattern_list, *args, **kwargs)
