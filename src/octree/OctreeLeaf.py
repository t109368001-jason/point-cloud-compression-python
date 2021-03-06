from typing import Optional

from src.octree import OctreeBranch
from src.octree.OctreeNode import OctreeNode


class OctreeLeaf(OctreeNode):
    def __init__(self, depth: int, origin: list, size: float, parent: OctreeBranch):
        super().__init__(depth, origin, size, parent)
        self.parent: OctreeBranch
        self.data = dict()

    def insert_point(self, point, store_in_leaf=False, *args, **kwargs) -> None:
        if store_in_leaf:
            if "points" not in self.data:
                self.data["points"] = []
            self.data["points"].append(point)

    def count_leaf(self, count_points=False, *args, **kwargs) -> int:
        if count_points:
            if "points" in self.data:
                return len(self.data["points"])
        return 1

    def serialize(self, *args, **kwargs) -> Optional[list]:
        # TODO
        return []

    def deserialize(self, bit_pattern_list: list, *args, **kwargs):
        # TODO
        return

    def to_points(self, *args, **kwargs):
        return [self.origin]

    def clear(self, *args, **kwargs):
        # TODO
        return
