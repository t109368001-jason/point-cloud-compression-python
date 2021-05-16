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

    def insert_point(self, point, store_in_leaf=False, selector=None, *args, **kwargs) -> None:
        super().insert_point(point, store_in_leaf, selector=selector)

    def count_leaf(self, count_points=False, selector=None, *args, **kwargs) -> int:
        return super().count_leaf(count_points, selector=selector)

    def init_children(self, index: int, selector=None, *args, **kwargs):
        new_bound = self.get_sub_bound(index)
        children: Union[OctreeLeaf, BufferedOctreeBranch]
        if self.depth == 1:
            children = OctreeLeaf(self.depth - 1, new_bound.origin, new_bound.size, self)
        else:
            children: BufferedOctreeBranch
            found_index = -1
            for i in range(self.get_buffer_size()):
                if i == selector:
                    continue
                children = self.get_children(index, selector=i)
                if children is not None:
                    found_index = i
                break
            if found_index < 0:
                children = BufferedOctreeBranch(self.depth - 1, new_bound.origin, new_bound.size, self,
                                                self.get_buffer_size())
            else:
                children = self.get_children(index, selector=found_index)
        self.set_children(index, children, selector=selector, *args, **kwargs)
        return children

    def has_children(self, selector=None, *args, **kwargs):
        return super().has_children(selector=selector)

    def get_children(self, index, selector=None, *args, **kwargs):
        selected: List[Optional[Union[OctreeLeaf, BufferedOctreeBranch]]]
        selected = self._children[selector]
        if selected is None:
            return None
        children: Optional[Union[OctreeLeaf, BufferedOctreeBranch]]
        children = selected[index]
        return children

    def get_children_list(self, selector=None, *args, **kwargs):
        selected = self._children[selector]
        return selected if selected is not None else [None] * 8

    def get_buffer_size(self):
        if self._children is None:
            return 0
        return len(self._children)

    def set_children(self, index, children: Optional[OctreeNode], selector=None, *args, **kwargs):
        selected: List[Optional[OctreeLeaf, OctreeBranch]]
        selected = self._children[selector]
        if selected is None:
            self._children[selector] = [None] * 8
            selected = self._children[selector]

        if self.depth == 1:
            children: List[OctreeLeaf]
            selected: List[OctreeLeaf]
            selected[index] = children
        else:
            children: List[OctreeBranch]
            selected: List[OctreeBranch]
            selected[index] = children

    def get_bit_pattern(self, selector=None, *args, **kwargs) -> np.uint8:
        return super().get_bit_pattern(selector=selector, *args, **kwargs)

    def serialize(self, selector=None, *args, **kwargs) -> Optional[list]:
        return super().serialize(selector=selector, *args, **kwargs)

    def deserialize(self, bit_pattern_list: list, selector=None, *args, **kwargs):
        super().deserialize(bit_pattern_list, selector=selector, *args, **kwargs)

    def buffer_diff(self, selector1, selector2, *args, **kwargs):
        if self.depth == 1:
            bit_pattern_list = self.get_bit_pattern_list(selector=selector1, *args, **kwargs)
            return np.sum(np.abs(bit_pattern_list - self.get_bit_pattern_list(selector=selector2, *args, **kwargs)))
        else:
            diff = 0
            for i in range(8):
                children1: Optional[BufferedOctreeBranch]
                children2: Optional[BufferedOctreeBranch]
                children1 = self.get_children(i, selector=selector1, *args, **kwargs)
                children2 = self.get_children(i, selector=selector2, *args, **kwargs)
                if children1 is not None:
                    diff += children1.buffer_diff(selector1, selector2, *args, **kwargs)
                elif children2 is not None:
                    diff += children2.buffer_diff(selector1, selector2, *args, **kwargs)
            return diff

    def diff(self, branch, selector, selector1, *args, **kwargs):
        branch: BufferedOctreeBranch
        if self.depth == 1:
            bit_pattern_list = branch.get_bit_pattern_list(selector=selector1, *args, **kwargs)
            return np.sum(np.abs(bit_pattern_list - self.get_bit_pattern_list(selector=selector, *args, **kwargs)))
        else:
            diff = 0
            for i in range(8):
                children1: Optional[BufferedOctreeBranch]
                children2: Optional[BufferedOctreeBranch]
                children1 = branch.get_children(i, selector=selector1, *args, **kwargs)
                children2 = self.get_children(i, selector=selector, *args, **kwargs)
                if children1 is None and children2 is not None:
                    diff += children2.count_leaf(selector=selector)
                elif children2 is None and children1 is not None:
                    diff += children1.count_leaf(selector=selector1)
                elif children1 is not None and children2 is not None:
                    diff += children2.diff(children1, selector=selector, selector1=selector1, *args, **kwargs)
            return diff

    def get_self_index(self, selector, *args, **kwargs):
        self.parent: BufferedOctreeBranch
        for index, children in enumerate(self.parent.get_children_list(selector=selector, *args, **kwargs)):
            if children is self:
                return index
        return ValueError("self is not parent's children")

    def find_min_diff_indices(self, branch, selector_i, selector_f) -> (int, list):
        branch: BufferedOctreeBranch
        if self.depth == branch.depth:
            diff = self.diff(branch, selector=selector_i, selector1=selector_f)
            return diff, []
        else:
            diff_indices_list = [
                children.find_min_diff_indices(branch, selector_i=selector_i, selector_f=selector_f)
                if children is not None else (pow(pow(2, self.depth), 3), [])
                for children in self.get_children_list(selector=selector_i)
            ]
            diff_list = [diff_indices[0] for diff_indices in diff_indices_list]
            indices_list = [diff_indices[1] for diff_indices in diff_indices_list]
            min_index = np.argmin(diff_list)
            indices = [min_index]
            indices.extend(indices_list[min_index])
            return diff_list[min_index], indices

    def motion_estimation(self, depth, selector_i, selector_f):
        indices = []
        if self.depth == depth:
            diff, indices = self.get_root_node().find_min_diff_indices(self,
                                                                       selector_i=selector_i,
                                                                       selector_f=selector_f)
            indices.extend(indices)
        else:
            for children in self.get_children_list(selector=selector_f):
                if children is not None:
                    indices_ = children.motion_estimation(depth, selector_i=selector_i, selector_f=selector_f)
                    indices.extend(indices_)
        return indices

    def get_by_indices(self, indices, depth, selector=None, *args, **kwargs):
        return super().get_by_indices(indices, depth, selector=selector, *args, **kwargs)

    def motion_compensation(self, depth, indices, selector_i, selector_p):
        if self.depth == depth:
            branch_i = self.get_root_node().get_by_indices(indices, depth=depth, selector=selector_i)
            self.set(branch_i, selector_i=selector_i, selector_p=selector_p)
        else:
            for children in self.get_children_list(selector=selector_p):
                if children is not None:
                    children.motion_compensation(depth, indices, selector_i, selector_p)

    def set(self, branch_i, selector_i, selector_p):
        if self.depth != branch_i.depth:
            raise ValueError("self.depth != branch.depth")
        if self.depth == 1:
            for index, children_i in enumerate(branch_i.get_children_list(selector=selector_i)):
                if children_i is not None:
                    new_children = OctreeLeaf(0, children_i.origin.copy(), children_i.size, self)
                    self.set_children(index, new_children, selector=selector_p)
        else:
            for index, children_i in enumerate(branch_i.get_children_list(selector=selector_i)):
                if children_i is not None:
                    self.set_children(index, children_i, selector=selector_p)
                    self.get_children(index, selector=selector_p).set(children_i,
                                                                      selector_i=selector_i,
                                                                      selector_p=selector_p)

    def to_points(self, selector, *args, **kwargs):
        return super().to_points(selector=selector, *args, **kwargs)

    def clear(self, selector, *args, **kwargs):
        for index, children in enumerate(self.get_children_list(selector=selector, *args, **kwargs)):
            if children is not None:
                children.clear(selector=selector, *args, **kwargs)
                empty = True
                for i in range(8):
                    if i == selector:
                        continue
                    other_children = self.get_children(index, selector=selector, *args, **kwargs)
                    if other_children is not None:
                        empty = False
                if empty:
                    children.parent = None
            self.set_children(index, None, selector=selector, *args, **kwargs)
