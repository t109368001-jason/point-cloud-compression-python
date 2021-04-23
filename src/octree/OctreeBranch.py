from typing import List, Optional

from src.octree.OctreeLeaf import OctreeLeaf
from src.octree.OctreeNode import OctreeNode


class OctreeBranch(OctreeNode):
    def __init__(self, depth: int, origin: list, size: float, parent):
        super().__init__(depth, origin, size, parent)
        self.parent: OctreeBranch
        if self.depth == 1:
            self.children: List[Optional[OctreeLeaf]] = [None] * 8
        else:
            self.children: List[Optional[OctreeBranch]] = [None] * 8

    def insert_point(self, point, store_in_leaf=False) -> None:
        index = self.get_index(point)
        if self.children[index] is None:
            self.init_children(index)
        if self.depth == 1:
            self.children: List[OctreeLeaf]
            self.children[index].insert_point(
                point,
                store_in_leaf,
            )
        else:
            self.children: List[OctreeBranch]
            self.children[index].insert_point(point, store_in_leaf)

    def count_leaf(self, count_points=False) -> int:
        if self.depth == 1:
            self.children: List[OctreeLeaf]
            return sum([
                children.count_leaf(count_points) for children in self.children
                if children is not None
            ])
        else:
            self.children: List[OctreeBranch]
            return sum([
                children.count_leaf(count_points) for children in self.children
                if children is not None
            ])

    def init_children(self, index: int) -> None:
        new_bound = self.get_sub_bound(index)
        if self.depth == 1:
            self.children: List[OctreeLeaf]
            self.children[index] = OctreeLeaf(self.depth - 1, new_bound.origin,
                                              new_bound.size, self)
        else:
            self.children: List[OctreeBranch]
            self.children[index] = OctreeBranch(self.depth - 1,
                                                new_bound.origin,
                                                new_bound.size, self)

    def set_children(self, index, children: OctreeNode):
        if self.depth == 1:
            children: List[OctreeLeaf]
            self.children: List[OctreeLeaf]
            self.children[index] = children
        else:
            children: List[OctreeBranch]
            self.children: List[OctreeBranch]
            self.children[index] = children
