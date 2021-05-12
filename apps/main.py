import logging
import logging.config
import os
import struct
import time

import numpy as np
import open3d as o3d
from scipy import spatial

from src.octree.BufferedOctree import BufferedOctree

CSV_HEADER = "Resolution,Depth,MSE FP, MSE PF,PSNR FP,PSNR PF,Encoding Time,Decoding Time,Bytes,Tree Depth,Points (I),Points (F),Tree Points (I),Tree Points (F),Tree Points (P)\n"
CSV_FMT = "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n"

# resolution_depth_dict = {
#     32: [3, 2, 1],
#     16: [4, 3, 2],
#     8: [5, 4, 3],
#     4: [6, 5, 4],
#     2: [7, 6, 5],
#     1: [8, 7, 6],
# }

# resolution_depth_dict = {
#     32: [3, 2, 1],
#     20: [3, 2, 1],
#     16: [4, 3, 2],
#     15: [4, 3, 2],
#     10: [5, 4, 3],
#     8: [5, 4, 3],
#     5: [6, 5, 4],
#     4: [6, 5, 4],
#     3: [7, 6, 5],
#     2: [7, 6, 5],
#     1.5: [8, 7, 6],
#     1: [8, 7, 6],
# }

resolution_depth_dict = {
    16: [1],
    15: [1],
    10: [2, 1],
    8: [2, 1],
    5: [3, 2, 1],
    4: [3, 2, 1],
    3: [4, 3, 2, 1],
    2: [4, 3, 2, 1],
    1.5: [5, 4, 3, 2, 1],
    1: [5, 4, 3, 2, 1],
}


def main():
    log_config_paths = ['logback.conf', 'apps/logback.conf']
    for log_config_path in log_config_paths:
        if os.path.exists(log_config_path):
            logging.config.fileConfig(log_config_path)
    logging.info("start")

    selected_buffer_i = 0
    selected_buffer_f = 1
    selected_buffer_p = 2

    point_cloud_i: o3d.geometry.PointCloud = o3d.io.read_point_cloud("../dataset/redandblack_vox10_1450.ply")
    point_cloud_f: o3d.geometry.PointCloud = o3d.io.read_point_cloud("../dataset/redandblack_vox10_1451.ply")
    points_i = np.asarray(point_cloud_i.points)
    points_f = np.asarray(point_cloud_f.points)

    max_value_range = np.max([
        np.max(points_i[:, 0]) - np.min(points_i[:, 0]),
        np.max(points_i[:, 1]) - np.min(points_i[:, 1]),
        np.max(points_i[:, 2]) - np.min(points_i[:, 2]),
        np.max(points_f[:, 0]) - np.min(points_f[:, 0]),
        np.max(points_f[:, 1]) - np.min(points_f[:, 1]),
        np.max(points_f[:, 2]) - np.min(points_f[:, 2]),
    ])

    points_i_len = points_i.shape[0]
    points_f_len = points_i.shape[0]

    if not os.path.exists('result.csv'):
        with open('result.csv', 'a') as result_csv:
            result_csv.write(CSV_HEADER)

    for resolution, depth_list in resolution_depth_dict.items():
        buffered_octree = BufferedOctree(resolution=resolution, buffer_size=3)
        buffered_octree.insert_points(points_i, selected_buffer=selected_buffer_i)
        buffered_octree.insert_points(points_f, selected_buffer=selected_buffer_f)
        logging.info(
            "buffered_octree.root_node(depth={}, origin={}, size={})".format(buffered_octree.root_node.depth,
                                                                             buffered_octree.root_node.origin,
                                                                             buffered_octree.root_node.size))
        for depth in depth_list:
            if buffered_octree.root_node.depth <= depth:
                continue
            buffered_octree.root_node.clear(selected_buffer=selected_buffer_p)
            encoding_start = time.time()
            bit_pattern_list, indices = buffered_octree.motion_estimation(depth, selected_buffer_i,
                                                                          selected_buffer_f)
            encoding_end = time.time()
            encoding_time = encoding_end - encoding_start

            bytes_len = save('output.txt', depth, bit_pattern_list, indices)

            depth, bit_pattern_list, indices = load('output.txt')

            decoding_start = time.time()
            buffered_octree.motion_compensation(depth, bit_pattern_list, indices, selected_buffer_i,
                                                selected_buffer_p)
            decoding_end = time.time()
            decoding_time = decoding_end - decoding_start

            points_i_t = buffered_octree.root_node.to_points(selected_buffer=selected_buffer_i)
            points_f_t = buffered_octree.root_node.to_points(selected_buffer=selected_buffer_f)
            points_p_t = buffered_octree.root_node.to_points(selected_buffer=selected_buffer_p)
            mse_fp, mse_pf = mse_calc(points_f, points_p_t)
            psnr_fp = 10 * np.log10(max_value_range * max_value_range / mse_fp)
            psnr_pf = 10 * np.log10(max_value_range * max_value_range / mse_pf)
            with open('result.csv', 'a') as result_csv:
                result_csv.write(
                    CSV_FMT.format(resolution, depth, mse_fp, mse_pf, psnr_fp, psnr_pf, encoding_time, decoding_time,
                                   bytes_len,
                                   buffered_octree.root_node.depth, points_i_len, points_f_len,
                                   len(points_i_t), len(points_f_t), len(points_p_t)))


def mse_calc(points1, points2):
    logging.info("start")

    tree2 = spatial.KDTree(points2)
    err1, _ = tree2.query(points1)

    tree1 = spatial.KDTree(points1)
    err2, _ = tree1.query(points2)

    return np.mean(np.square(err1)), np.mean(np.square(err2))


def save(filename, depth, bit_pattern_list, indices):
    indices = np.array(indices).astype(np.uint8).reshape(len(indices), 1)
    indices_byte = np.unpackbits(indices, axis=1, bitorder='little')
    indices_byte = np.array([index[:3] for index in indices_byte])
    indices_byte = indices_byte.flatten()
    indices_byte_padded = np.append(indices_byte, np.zeros(
        shape=int(np.ceil(indices_byte.shape[0] / 8) * 8 - indices_byte.shape[0]))).astype(np.uint8)
    indices_byte_padded = indices_byte_padded.reshape(int(indices_byte_padded.shape[0] / 8), 8)
    indices_uint8 = np.packbits(indices_byte_padded, axis=1, bitorder='little').astype(np.uint8)
    indices_uint8 = indices_uint8.reshape(indices_uint8.shape[0])
    bytes_len = 0
    with open(filename, 'wb') as file:
        file.write(struct.pack('B', np.uint8(depth)))
        bytes_len += 1
        file.write(struct.pack('i', len(bit_pattern_list)))
        bytes_len += 4
        for bit_pattern in bit_pattern_list:
            file.write(struct.pack('B', bit_pattern))
            bytes_len += 1
        for index in indices_uint8:
            file.write(struct.pack('B', index))
            bytes_len += 1

    return bytes_len


def load(filename):
    with open(filename, 'rb') as file:
        depth = struct.unpack('B', file.read(1))[0]
        bit_pattern_list_len = struct.unpack('i', file.read(4))[0]
        byte_list = file.read()
        bit_pattern_list = [np.uint8(byte) for byte in byte_list[:bit_pattern_list_len]]
        indices_uint8 = [np.uint8(byte) for byte in byte_list[bit_pattern_list_len:]]

    indices_byte_padded = np.unpackbits(np.array(indices_uint8).astype(np.uint8), axis=0, bitorder='little')
    indices_byte = indices_byte_padded[:int(np.floor(len(indices_byte_padded) / 3) * 3)]
    indices_byte = indices_byte.reshape(int(indices_byte.shape[0] / 3), 3)
    indices = np.packbits(indices_byte, axis=1, bitorder='little')
    indices = indices.reshape(indices.shape[0])
    indices = list(indices)
    return depth, bit_pattern_list, indices


if __name__ == '__main__':
    main()
