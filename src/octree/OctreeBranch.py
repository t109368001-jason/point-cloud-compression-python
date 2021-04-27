from typing import List, Optional, Union

import numpy as np

from src.octree.OctreeLeaf import OctreeLeaf
from src.octree.OctreeNode import OctreeNode


class OctreeBranch(OctreeNode):
    def __init__(self, depth: int, origin: list, size: float, parent):
        super().__init__(depth, origin, size, parent)
        self.parent: OctreeBranch
        self._children: List[Optional[Union[OctreeLeaf, OctreeBranch]]] = [None] * 8

    def insert_point(self, point, store_in_leaf=False, *args, **kwargs) -> None:
        index = self.get_index(point)
        if self.get_children(index, *args, **kwargs) is None:
            children = self.init_children(index, *args, **kwargs)
        else:
            children = self.get_children(index, *args, **kwargs)
        if self.depth == 1:
            children: OctreeLeaf
            children.insert_point(point, store_in_leaf, *args, **kwargs)
        else:
            children: Optional[OctreeBranch]
            children.insert_point(point, store_in_leaf, *args, **kwargs)

    def count_leaf(self, count_points=False, *args, **kwargs) -> int:
        count = 0
        for children in self.get_children_list(*args, **kwargs):
            if children is not None:
                if self.depth == 1:
                    children: OctreeLeaf
                    count += children.count_leaf(count_points, *args, **kwargs)
                else:
                    children: OctreeBranch
                    count += children.count_leaf(count_points, *args, **kwargs)
        return count

    def init_children(self, index: int, *args, **kwargs):
        new_bound = self.get_sub_bound(index)
        children: Union[OctreeLeaf, OctreeBranch]
        if self.depth == 1:
            children = OctreeLeaf(self.depth - 1, new_bound.origin, new_bound.size, self)
        else:
            children = OctreeBranch(self.depth - 1, new_bound.origin, new_bound.size, self)
        self.set_children(index, children, *args, **kwargs)
        return children

    def has_children(self, *args, **kwargs):
        for children in self.get_children_list(*args, **kwargs):
            if children is not None:
                return True
        return False

    # noinspection PyUnusedLocal
    def get_children(self, index, *args, **kwargs):
        children: Union[OctreeLeaf, OctreeBranch]
        children = self._children[index]
        return children

    # noinspection PyUnusedLocal
    def get_children_list(self, *args, **kwargs):
        children: List[Union[OctreeLeaf, OctreeBranch]]
        children = self._children
        return children

    # noinspection PyUnusedLocal
    def set_children(self, index, children: OctreeNode, *args, **kwargs):
        if self.depth == 1:
            children: List[OctreeLeaf]
            self._children: List[OctreeLeaf]
            self._children[index] = children
        else:
            children: List[OctreeBranch]
            self._children: List[OctreeBranch]
            self._children[index] = children

    def get_bit_pattern(self, *args, **kwargs) -> np.uint8:
        bit_pattern = np.array(self.get_children_list(*args, **kwargs))
        bit_pattern = np.not_equal(bit_pattern, None)
        bit_pattern = np.packbits(bit_pattern, axis=0, bitorder='little').astype(np.uint8)
        return bit_pattern[0]

    def serialize(self, *args, **kwargs) -> Optional[list]:
        bit_pattern_list = list()
        bit_pattern_list.append(self.get_bit_pattern(*args, **kwargs))
        for children in self.get_children_list(*args, **kwargs):
            if children is not None:
                if self.depth == 1:
                    children: OctreeLeaf
                    bit_pattern_list.extend(children.serialize(*args, **kwargs))
                else:
                    children: OctreeBranch
                    bit_pattern_list.extend(children.serialize(*args, **kwargs))
        return bit_pattern_list

    def deserialize(self, bit_pattern_list: list, *args, **kwargs):
        bit_pattern = np.unpackbits(bit_pattern_list.pop(0), axis=0, bitorder='little')
        for index, bit in enumerate(bit_pattern):
            if bit == 1:
                self.init_children(index, *args, **kwargs)
                children = self.get_children(index, *args, **kwargs)
                if self.depth == 1:
                    children: OctreeLeaf
                    children.deserialize(bit_pattern_list, *args, **kwargs)
                else:
                    children: OctreeBranch
                    children.deserialize(bit_pattern_list, *args, **kwargs)
