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

    def set_children(self, index, children: Optional[OctreeNode], selected_buffer=None, *args, **kwargs):
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
        return super().get_bit_pattern(selected_buffer=selected_buffer, *args, **kwargs)

    def serialize(self, selected_buffer=None, *args, **kwargs) -> Optional[list]:
        return super().serialize(selected_buffer=selected_buffer, *args, **kwargs)

    def deserialize(self, bit_pattern_list: list, selected_buffer=None, *args, **kwargs):
        super().deserialize(bit_pattern_list, selected_buffer=selected_buffer, *args, **kwargs)

    def buffer_diff(self, selected_buffer1, selected_buffer2, *args, **kwargs):
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
                    diff += children1.buffer_diff(selected_buffer1, selected_buffer2, *args, **kwargs)
                elif children2 is not None:
                    diff += children2.buffer_diff(selected_buffer1, selected_buffer2, *args, **kwargs)
            return diff

    def diff(self, branch, selected_buffer, selected_buffer1, *args, **kwargs):
        branch: BufferedOctreeBranch
        if self.depth == 1:
            bit_pattern_list = branch.get_bit_pattern_list(selected_buffer=selected_buffer1, *args, **kwargs)
            return np.sum(
                np.abs(bit_pattern_list - self.get_bit_pattern_list(selected_buffer=selected_buffer, *args, **kwargs)))
        else:
            diff = 0
            for i in range(8):
                children1: Optional[BufferedOctreeBranch]
                children2: Optional[BufferedOctreeBranch]
                children1 = branch.get_children(i, selected_buffer=selected_buffer1, *args, **kwargs)
                children2 = self.get_children(i, selected_buffer=selected_buffer, *args, **kwargs)
                if children1 is None and children2 is not None:
                    diff += children2.count_leaf(selected_buffer=selected_buffer)
                elif children2 is None and children1 is not None:
                    diff += children1.count_leaf(selected_buffer=selected_buffer1)
                elif children1 is not None and children2 is not None:
                    diff += children2.diff(children1, selected_buffer=selected_buffer,
                                           selected_buffer1=selected_buffer1, *args, **kwargs)
            return diff

    def get_self_index(self, selected_buffer, *args, **kwargs):
        self.parent: BufferedOctreeBranch
        for index, children in enumerate(self.parent.get_children_list(selected_buffer=selected_buffer, *args,
                                                                       **kwargs)):
            if children is self:
                return index
        return ValueError("self is not parent's children")

    def find_min_diff_indices(self, branch, selected_buffer_i, selected_buffer_f) -> (int, list):
        branch: BufferedOctreeBranch
        if self.depth == branch.depth:
            diff = self.diff(branch, selected_buffer=selected_buffer_i, selected_buffer1=selected_buffer_f)
            return diff, []
        else:
            diff_indices_list = [
                children.find_min_diff_indices(branch, selected_buffer_i=selected_buffer_i,
                                               selected_buffer_f=selected_buffer_f) if children is not None else (
                    pow(pow(2, self.depth), 3), [])
                for
                children in self.get_children_list(selected_buffer=selected_buffer_i)]
            diff_list = [diff_indices[0] for diff_indices in diff_indices_list]
            indices_list = [diff_indices[1] for diff_indices in diff_indices_list]
            min_index = np.argmin(diff_list)
            indices = [min_index]
            indices.extend(indices_list[min_index])
            return diff_list[min_index], indices

    def get_root_node(self):
        branch = self
        while branch.parent is not None:
            branch = branch.parent
        return branch

    def motion_estimation(self, depth, selected_buffer_i, selected_buffer_f):
        indices = []
        if self.depth == depth:
            diff, indices = self.get_root_node().find_min_diff_indices(self, selected_buffer_i=selected_buffer_i,
                                                                       selected_buffer_f=selected_buffer_f)
            indices.extend(indices)
        else:
            for children in self.get_children_list(selected_buffer=selected_buffer_f):
                if children is not None:
                    indices_list_ = children.motion_estimation(depth, selected_buffer_i=selected_buffer_i,
                                                               selected_buffer_f=selected_buffer_f)
                    indices.extend(indices_list_)
        return indices

    def get_by_indices(self, indices, depth, selected_buffer):
        if len(indices) == 0:
            return self
        if self.depth == depth:
            return self
        index = indices.pop(0)
        return self.get_children(index, selected_buffer=selected_buffer).get_by_indices(indices=indices, depth=depth,
                                                                                        selected_buffer=selected_buffer)

    def motion_compensation(self, depth, indices, selected_buffer_i, selected_buffer_p):
        if self.depth == depth:
            branch_i = self.get_root_node().get_by_indices(indices, depth=depth, selected_buffer=selected_buffer_i)
            self.set(branch_i, selected_buffer_i=selected_buffer_i, selected_buffer_p=selected_buffer_p)
        else:
            for children in self.get_children_list(selected_buffer=selected_buffer_p):
                if children is not None:
                    children.motion_compensation(depth, indices, selected_buffer_i, selected_buffer_p)

    def set(self, branch_i, selected_buffer_i, selected_buffer_p):
        if self.depth != branch_i.depth:
            raise ValueError("self.depth != branch.depth")
        if self.depth == 1:
            for index, children_i in enumerate(branch_i.get_children_list(selected_buffer=selected_buffer_i)):
                if children_i is not None:
                    new_children = OctreeLeaf(0, children_i.origin.copy(), children_i.size, self)
                    self.set_children(index, new_children, selected_buffer=selected_buffer_p)
        else:
            for index, children_i in enumerate(branch_i.get_children_list(selected_buffer=selected_buffer_i)):
                if children_i is not None:
                    self.set_children(index, children_i, selected_buffer=selected_buffer_p)
                    self.get_children(index, selected_buffer=selected_buffer_p).set(children_i,
                                                                                    selected_buffer_i=selected_buffer_i,
                                                                                    selected_buffer_p=selected_buffer_p)

    def to_points(self, selected_buffer, *args, **kwargs):
        return super().to_points(selected_buffer=selected_buffer, *args, **kwargs)

    def clear(self, selected_buffer, *args, **kwargs):
        for index, children in enumerate(self.get_children_list(selected_buffer=selected_buffer, *args, **kwargs)):
            if children is not None:
                children.clear(selected_buffer=selected_buffer, *args, **kwargs)
                empty = True
                for i in range(8):
                    if i == selected_buffer:
                        continue
                    other_children = self.get_children(index, selected_buffer=selected_buffer, *args, **kwargs)
                    if other_children is not None:
                        empty = False
                if empty:
                    children.parent = None
            self.set_children(index, None, selected_buffer=selected_buffer, *args, **kwargs)
