from typing import Optional, List, Union

import numpy as np

from src.octree.OctreeBranch import OctreeBranch
from src.octree.OctreeLeaf import OctreeLeaf
from src.octree.OctreeNode import OctreeNode


class BufferedOctreeBranch(OctreeBranch):
    def __init__(self, depth: int, origin: list, size: float, parent, buffer_size):
        super(OctreeBranch, self).__init__(depth, origin, size, parent)
        self.parent: BufferedOctreeBranch
        self._children: List[List[Optional[Union[OctreeLeaf, BufferedOctreeBranch]]]] = [None] * buffer_size

    def insert_point(self, point, store_in_leaf=False, selected_buffer=None, *args, **kwargs) -> None:
        super().insert_point(point, store_in_leaf, selected_buffer=selected_buffer)

    def count_leaf(self, count_points=False, selected_buffer=None, *args, **kwargs) -> int:
        return super().count_leaf(count_points, selected_buffer=selected_buffer)

    def init_children(self, index: int, selected_buffer=None, *args, **kwargs):
        new_bound = self.get_sub_bound(index)
        children: Union[OctreeLeaf, BufferedOctreeBranch]
        if self.depth == 1:
            children = OctreeLeaf(self.depth - 1, new_bound.origin, new_bound.size, self)
        else:
            children: BufferedOctreeBranch
            found_index = -1
            for i in range(self.get_buffer_size()):
                if i == selected_buffer:
                    continue
                children = self.get_children(index, selected_buffer=i)
                if children is not None:
                    found_index = i
                break
            if found_index < 0:
                children = BufferedOctreeBranch(self.depth - 1, new_bound.origin, new_bound.size, self,
                                                self.get_buffer_size())
            else:
                children = self.get_children(index, selected_buffer=found_index)
        self.set_children(index, children, selected_buffer=selected_buffer, *args, **kwargs)
        return children

    def has_children(self, selected_buffer=None, *args, **kwargs):
        return super().has_children(selected_buffer=selected_buffer)

    def get_children(self, index, selected_buffer=None, *args, **kwargs):
        selected: List[Optional[Union[OctreeLeaf, BufferedOctreeBranch]]]
        selected = self._children[selected_buffer]
        if selected is None:
            return None
        children: Optional[Union[OctreeLeaf, BufferedOctreeBranch]]
        children = selected[index]
        return children

    def get_children_list(self, selected_buffer=None, *args, **kwargs):
        selected = self._children[selected_buffer]
        return selected if selected is not None else [None] * 8

    def get_buffer_size(self):
        if self._children is None:
            return 0
        return len(self._children)

    def set_children(self, index, children: OctreeNode, selected_buffer=None, *args, **kwargs):
        selected: List[Optional[OctreeLeaf, OctreeBranch]]
        selected = self._children[selected_buffer]
        if selected is None:
            self._children[selected_buffer] = [None] * 8
            selected = self._children[selected_buffer]

        if self.depth == 1:
            children: List[OctreeLeaf]
            selected: List[OctreeLeaf]
            selected[index] = children
        else:
            children: List[OctreeBranch]
            selected: List[OctreeBranch]
            selected[index] = children

    def get_bit_pattern(self, selected_buffer=None, *args, **kwargs) -> np.uint8:
        return super().get_bit_pattern(selected_buffer=selected_buffer)

    def serialize(self, selected_buffer=None, *args, **kwargs) -> Optional[list]:
        return super().serialize(selected_buffer=selected_buffer)

    def deserialize(self, bit_pattern_list: list, selected_buffer=None, *args, **kwargs):
        super().deserialize(bit_pattern_list, selected_buffer=selected_buffer)

    def diff(self, selected_buffer1, selected_buffer2, *args, **kwargs):
        other: Optional[BufferedOctreeBranch]
        if self.depth == 1:
            bit_pattern_list = self.get_bit_pattern_list(selected_buffer=selected_buffer1, *args, **kwargs)
            return np.sum(
                np.abs(bit_pattern_list - self.get_bit_pattern_list(selected_buffer=selected_buffer2, *args, **kwargs)))
        else:
            diff = 0
            for i in range(8):
                children1: Optional[BufferedOctreeBranch]
                children2: Optional[BufferedOctreeBranch]
                children1 = self.get_children(i, selected_buffer=selected_buffer1, *args, **kwargs)
                children2 = self.get_children(i, selected_buffer=selected_buffer2, *args, **kwargs)
                if children1 is not None:
                    diff += children1.diff(selected_buffer1, selected_buffer2, *args, **kwargs)
                elif children2 is not None:
                    diff += children2.diff(selected_buffer1, selected_buffer2, *args, **kwargs)
            return diff
