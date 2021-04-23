from src.octree.Bound import Bound


class OctreeNode(Bound):
    def __init__(self, depth: int, origin: list, size: float, parent):
        super().__init__(origin, size)
        self.depth = depth
        self.parent: OctreeNode = parent

    def insert_point(self, point, store_in_leaf=False) -> None:
        raise NotImplementedError()

    def count_leaf(self, count_points=False) -> int:
        raise NotImplementedError()
