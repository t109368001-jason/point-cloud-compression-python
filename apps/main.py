import time

import numpy as np
import open3d as o3d

from src.octree.Octree import Octree

if __name__ == '__main__':
    point_cloud: o3d.geometry.PointCloud = o3d.io.read_point_cloud(
        "../dataset/redandblack_vox10_1450.ply")
    points = np.asarray(point_cloud.points)
    octree = Octree(resolution=1)
    insert_start = time.time()
    octree.insert_points(points, True)
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
    print(octree.root_node.count_leaf())
    print(new_octree.root_node.count_leaf())
