import time

import numpy as np
import open3d as o3d

from src.octree.BufferedOctree import BufferedOctree
from src.octree.Octree import Octree

if __name__ == '__main__':
    point_cloud: o3d.geometry.PointCloud = o3d.io.read_point_cloud("../dataset/redandblack_vox10_1450.ply")
    points = np.asarray(point_cloud.points)
    octree = Octree(resolution=1)
    insert_start = time.time()
    octree.insert_points(points[:1000], True)
    insert_end = time.time()
    print("insert={}".format(insert_end - insert_start))
    serialize_start = time.time()
    bit_pattern_list, resolution, depth, origin, size = octree.serialize()
    serialize_end = time.time()
    print("serialize={}".format(serialize_end - serialize_start))
    new_octree = Octree(resolution=1)
    deserialize_start = time.time()
    new_octree.deserialize(bit_pattern_list, resolution, depth, origin, size)
    deserialize_end = time.time()
    print("deserialize={}".format(deserialize_end - deserialize_start))
    print("octree.root_node.count_leaf={}".format(octree.root_node.count_leaf()))
    print("new_octree.root_node.count_leaf={}".format(new_octree.root_node.count_leaf()))

    buffered_octree = BufferedOctree(resolution=1, buffer_size=2)
    buffered_octree.insert_points(points[:1000], selected_buffer=0)
    buffered_octree.insert_points(points[:2000], selected_buffer=1)
    print("buffered_octree.root_node.count_leaf(selected_buffer=0)={}".format(
        buffered_octree.root_node.count_leaf(selected_buffer=0)))
    print("buffered_octree.root_node.count_leaf(selected_buffer=1)={}".format(
        buffered_octree.root_node.count_leaf(selected_buffer=1)))

    diff = buffered_octree.root_node.diff(0, 1)
    print("buffered_octree.diff(0, 1)={}".format(diff))
