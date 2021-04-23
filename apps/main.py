import numpy as np
import open3d as o3d

from src.octree.Octree import Octree

if __name__ == '__main__':
    point_cloud: o3d.geometry.PointCloud = o3d.io.read_point_cloud(
        "../dataset/redandblack_vox10_1450.ply")
    points = np.asarray(point_cloud.points)
    octree = Octree(resolution=1)
    octree.insert_points(points, True)

    print(octree.root_node.count_leaf())
    print(octree.root_node.count_leaf(count_points=True))
