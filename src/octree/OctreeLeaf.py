from src.octree import OctreeBranch
from src.octree.OctreeNode import OctreeNode


class OctreeLeaf(OctreeNode):
    def __init__(self, depth: int, origin: list, size: float,
                 parent: OctreeBranch):
        super().__init__(depth, origin, size, parent)
        self.parent: OctreeBranch
        self.data = dict()

    def insert_point(self, point, store_in_leaf=False) -> None:
        if store_in_leaf:
            if "points" not in self.data:
                self.data["points"] = []
            self.data["points"].append(point)

    def count_leaf(self, count_points=False) -> int:
        if count_points:
            if "points" in self.data:
                return len(self.data["points"])
        return 1
