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

    def get_children(self, index, *args, **kwargs):
        children: Union[OctreeLeaf, OctreeBranch]
        children = self._children[index]
        return children

    def get_children_list(self, *args, **kwargs):
        children: List[Union[OctreeLeaf, OctreeBranch]]
        children = self._children
        return children

    def set_children(self, index, children: Optional[OctreeNode], *args, **kwargs):
        if self.depth == 1:
            children: List[OctreeLeaf]
            self._children: List[OctreeLeaf]
            self._children[index] = children
        else:
            children: List[OctreeBranch]
            self._children: List[OctreeBranch]
            self._children[index] = children

    def get_bit_pattern_list(self, *args, **kwargs) -> np.uint8:
        bit_pattern_list = np.array(self.get_children_list(*args, **kwargs))
        bit_pattern_list = np.not_equal(bit_pattern_list, None) * 1
        return bit_pattern_list

    def get_bit_pattern(self, *args, **kwargs) -> np.uint8:
        bit_pattern_list = self.get_bit_pattern_list(*args, **kwargs)
        bit_pattern = np.packbits(bit_pattern_list, axis=0, bitorder='little').astype(np.uint8)
        return bit_pattern[0]

    def serialize(self, depth=None, *args, **kwargs) -> Optional[list]:
        bit_pattern_list = list()
        bit_pattern_list.append(self.get_bit_pattern(*args, **kwargs))
        if (depth is not None) and (self.depth <= depth):
            return bit_pattern_list
        for children in self.get_children_list(*args, **kwargs):
            if children is not None:
                if self.depth == 1:
                    children: OctreeLeaf
                    bit_pattern_list.extend(children.serialize(depth=depth, *args, **kwargs))
                else:
                    children: OctreeBranch
                    bit_pattern_list.extend(children.serialize(depth=depth, *args, **kwargs))
        return bit_pattern_list

    def deserialize(self, bit_pattern_list: list, depth=None, *args, **kwargs):
        bit_pattern = np.unpackbits(np.uint8(bit_pattern_list.pop(0)), axis=0, bitorder='little')
        if (depth is not None) and (self.depth < depth):
            return
        for index, bit in enumerate(bit_pattern):
            if bit == 1:
                self.init_children(index, *args, **kwargs)
                if (depth is not None) and (self.depth <= depth):
                    continue
                children = self.get_children(index, *args, **kwargs)
                if self.depth == 1:
                    children: OctreeLeaf
                    children.deserialize(bit_pattern_list, depth=depth, *args, **kwargs)
                else:
                    children: OctreeBranch
                    children.deserialize(bit_pattern_list, depth=depth, *args, **kwargs)

    def to_points(self, *args, **kwargs):
        points = []
        for children in self.get_children_list(*args, **kwargs):
            if children is not None:
                children_points = children.to_points(*args, **kwargs)
                points.extend(children_points)
        return points

    def clear(self, *args, **kwargs):
        for index, children in enumerate(self.get_children_list(*args, **kwargs)):
            if children is not None:
                children.clear(*args, **kwargs)
                children.parent = None
            self.set_children(index, None, *args, **kwargs)

    def get_by_indices(self, indices, depth, *args, **kwargs):
        if len(indices) == 0:
            return self
        if self.depth == depth:
            return self
        index = indices.pop(0)
        return self.get_children(index, *args, **kwargs).get_by_indices(indices, depth, *args, **kwargs)
