import sys

import cv2
from pathlib import Path
import numpy as np
import math

# from utils import get_macro_blocks_from_frame

# folder_path = 'video_rgb\\SAL_490_270_437'
# width = 490
# height = 270
macro_block_size = 16
np.set_printoptions(threshold=sys.maxsize)


def _mv_to_deg(motion_vector):
    return np.angle(complex(motion_vector[0], motion_vector[1]), deg=True)


def _to_yuv_frame(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)


def _get_macro_blocks_from_rgb_frame(frame):
    frame_h, frame_w, pixel_unit_size = frame.shape
    macro_block_matrix_width = math.floor(frame_w / macro_block_size)
    macro_block_matrix_height = math.floor(frame_h / macro_block_size)
    macro_blocks = np.empty((macro_block_matrix_height, macro_block_matrix_width, 16, 16, 3))
    for i in range(macro_block_matrix_height):
        for j in range(macro_block_matrix_width):
            macro_blocks[i][j] = _get_rgb_marco_block_from_rgb_frame_with_left_top_corner(
                i * macro_block_size, j * macro_block_size, frame)
    return macro_blocks


def _get_macro_blocks_from_yuv_frame(frame):
    frame_h, frame_w, pixel_unit_size = frame.shape
    macro_block_matrix_width = math.floor(frame_w / macro_block_size)
    macro_block_matrix_height = math.floor(frame_h / macro_block_size)
    macro_blocks = np.empty((macro_block_matrix_height, macro_block_matrix_width, 16, 16))
    for i in range(macro_block_matrix_height):
        for j in range(macro_block_matrix_width):
            macro_blocks[i][j] = _get_y_marco_block_from_yuv_frame_with_left_top_corner(
                i * macro_block_size, j * macro_block_size, frame)
    return macro_blocks


def _get_y_marco_block_from_yuv_frame_with_left_top_corner(x_h, y_w, yuv_frame):
    return yuv_frame[np.ix_(
        list(range(x_h, x_h + macro_block_size)),
        list(range(y_w, y_w + macro_block_size)),
        [0]
    )].reshape(16, 16)


def _get_rgb_marco_block_from_rgb_frame_with_left_top_corner(x_h, y_w, rgb_frame):
    return rgb_frame[np.ix_(
        list(range(x_h, x_h + macro_block_size)),
        list(range(y_w, y_w + macro_block_size)),
        [0, 1, 2]
    )].reshape(16, 16, 3)


def _get_mad_of_two_marco_block(macro1, macro2):
    return np.absolute(np.subtract(np.array(macro1), macro2)).mean()


def _resize(frame, new_size_h_w):
    return cv2.resize(frame, (new_size_h_w[1], new_size_h_w[0]), interpolation=cv2.INTER_AREA)


class RGBImage:

    def __init__(self, cur_frame: np.array, pre_frame: np.array, new_size_h_w=None):
        if new_size_h_w:
            self.cur_frame = _resize(cur_frame, new_size_h_w)
            self.pre_frame = _resize(pre_frame, new_size_h_w)
        else:
            self.cur_frame = cur_frame
            self.pre_frame = pre_frame
        self.frame_h, self.frame_w, self.pixel_unit_size = self.cur_frame.shape
        #print(self.cur_frame.shape)
        self.cur_yuv_frame = _to_yuv_frame(self.cur_frame)
        self.pre_yuv_frame = _to_yuv_frame(self.pre_frame)

        # -------------------
        self.macro_block_matrix_width = math.floor(self.frame_w / macro_block_size)
        self.macro_block_matrix_height = math.floor(self.frame_h / macro_block_size)
        self.block_arr = self._create_block()
        # -------------------

        self.rgb_macro_blocks = _get_macro_blocks_from_rgb_frame(self.cur_frame)
        self.macro_block_mappings = self._build_marco_block_mapping_between_frame()

        self.motion_vectors_allocated_as_blocks = np.array(list(map(lambda x: x[2], self.macro_block_mappings))) \
            .reshape(self.macro_block_matrix_height, self.macro_block_matrix_width, 2)
        self.object_motion_vector_list = self._reduce_motion_vectors(self.motion_vectors_allocated_as_blocks)
        self.macro_block_labels_allocated_as_blocks = np.array(
            list(map(lambda x: list(self.object_motion_vector_list.keys()).index(_mv_to_deg(x[2])),
                     self.macro_block_mappings))) \
            .reshape(self.macro_block_matrix_height, self.macro_block_matrix_width)

    def _reduce_motion_vectors(self, motion_vectors_allocated_as_blocks):
        res = {}
        motion_vectors = [item for row in motion_vectors_allocated_as_blocks for item in row]
        for motion_vector in motion_vectors:
            # print(motion_vector)
            motion_vector_tuple = (motion_vector[0], motion_vector[1])
            motion_vector_angle = _mv_to_deg(motion_vector)
            motion_vector_key = motion_vector_angle

            if motion_vector_key not in res.keys():
                res[motion_vector_key] = {
                    "degree": motion_vector_angle,
                    "total": 0,
                    "sub_mvs": {
                    }
                }
            res[motion_vector_key]["total"] += 1

            if motion_vector_tuple not in res[motion_vector_key]["sub_mvs"]:
                res[motion_vector_key]["sub_mvs"][motion_vector_tuple] = 0
            res[motion_vector_key]["sub_mvs"][motion_vector_tuple] += 1
        return dict(sorted(res.items(), key=lambda item: -item[1]["total"]))

    # def test(self, x, y):
    #     return _get_marco_block_with_left_top_corner(x, y);

    def _build_marco_block_mapping_between_frame(self):
        cur_macro_block_matrix = _get_macro_blocks_from_yuv_frame(self.cur_yuv_frame)
        res = []
        for cur_i in range(len(cur_macro_block_matrix)):
            for cur_j in range(len(cur_macro_block_matrix[0])):
                cur_macro_block = cur_macro_block_matrix[cur_i][cur_j]
                cur_macro_block_left_top_corner = (cur_i * macro_block_size, cur_j * macro_block_size)
                square_area_coords = self._get_square_area_coords(cur_macro_block_left_top_corner, x_h_range=[-6, 6],
                                                                  y_w_range=[-6, 6])
                mad = sys.maxsize
                ref_macro_block = None
                ref_macro_block_left_top_corner = (cur_i * macro_block_size, cur_j * macro_block_size)
                for coord in square_area_coords:
                    pre_macro_block = _get_y_marco_block_from_yuv_frame_with_left_top_corner(coord[0], coord[1],
                                                                                             self.pre_yuv_frame)
                    new_mad = _get_mad_of_two_marco_block(cur_macro_block, pre_macro_block)
                    if mad > new_mad:
                        mad = new_mad
                        ref_macro_block_left_top_corner = coord
                        ref_macro_block = pre_macro_block
                res.append((cur_macro_block, ref_macro_block,
                            (cur_macro_block_left_top_corner[0] - ref_macro_block_left_top_corner[0],
                             cur_macro_block_left_top_corner[1] - ref_macro_block_left_top_corner[1])))
                # print(f"macro block({cur_i},{cur_j})->done")
        return res

    def _get_square_area_coords(self, center_point, x_h_range, y_w_range):
        center_x = center_point[0]
        center_y = center_point[1]
        res = []
        for offset_x in range(x_h_range[0], x_h_range[1] + 1):
            for offset_y in range(y_w_range[0], y_w_range[1] + 1):
                if 0 <= (offset_x + center_x) < self.frame_h - macro_block_size \
                        and 0 <= (offset_y + center_y) < self.frame_w - macro_block_size:
                    res.append((offset_x + center_x, offset_y + center_y))
        if (center_x, center_y) in res:
            res.remove((center_x, center_y))
        return res

    def display(self, flag):
        """
        :param flag: flag=0 if bgr_arr; else block_arr
        :return:
        """
        if flag == 0:
            cv2.imshow('image', self.cur_frame)
        else:
            arr = cv2.resize(self.block_arr, None, fx=macro_block_size, fy=macro_block_size)
            cv2.imshow('image', arr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_background(self):
        def mv_is_foreground(mv):
            return abs(_mv_to_deg(mv) - list(self.object_motion_vector_list.keys())[0]) > 30
                   # and (
                   #      mv[0] ** 2 + mv[1] ** 2) > 1

        background_frame = self.cur_frame
        # self.cur_frame_mar
        for i in range(len(self.macro_block_labels_allocated_as_blocks)):
            for j in range(len(self.macro_block_labels_allocated_as_blocks[0])):
                # print(self.object_motion_vector_list.items()[self.macro_block_labels_allocated_as_blocks[i][j]][1]>300)

                # if list(self.object_motion_vector_list.items())[self.macro_block_labels_allocated_as_blocks[i][j]][
                #     1] < 300:
                if mv_is_foreground(self.macro_block_mappings[i * self.macro_block_matrix_width + j][2]):
                    # if abs(_mv_to_deg(self.macro_block_mappings[i*self.macro_block_matrix_width+j][2])-list(self.object_motion_vector_list.keys())[0])>20:
                    # if self.macro_block_labels_allocated_as_blocks[i][j] != 0:
                    for k in range(16):
                        for l in range(16):
                            background_frame[i * macro_block_size + k][j * macro_block_size + l] = np.array([0, 0, 0])

        cv2.imshow('image', background_frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def _create_block(self):
        block_y_arr = np.empty((self.macro_block_matrix_height, self.macro_block_matrix_width, 3), np.uint8)
        yuv_arr = cv2.cvtColor(self.cur_frame, cv2.COLOR_BGR2YUV)
        for h in range(self.macro_block_matrix_height):
            for w in range(self.macro_block_matrix_width):
                val = np.average(yuv_arr[h * macro_block_size:(h + 1) * macro_block_size,
                                 w * macro_block_size:(w + 1) * macro_block_size, 0]
                                 , axis=(0, 1))
                block_y_arr[h, w] = val
        return block_y_arr

    def get_macro_block_label(self) -> np.array:
        """

        block_label[i,j] = label of (i,j)
        label = 0 if background, label != 0 if foreground
        :return: block_label: numpy array of shape (height/block_size, width/block_size)
        """
        # block_label = np.empty((width, height))
        return self.macro_block_labels_allocated_as_blocks

    def get_macro_block(self, i, j) -> np.array:
        """

        :param i: index in height
        :param j: index in width
        :return: numpy array of block_size * block_size
        """
        # print(self.macro_block_mappings[i*self.macro_block_matrix_width+j][1])
        return self.rgb_macro_blocks[i][j]

    def get_original_bgr_array(self) -> np.array:
        """
        :return:
        """
        return self.cur_frame

    def get_motion_vector(self, label) -> '(dy, dx)':
        """
        Motion vector based on previous frame
        :param label: label retrieve by get_macro_block_label
        :return: vertical motion in pixels, horizontal motion in pixels
        """
        return list(self.object_motion_vector_list.keys())[label]

    def get_eight_point(self):
        """

        point: (y,x)
        :return: src point: point1, point2, point3, point4
                dst point: point5, point6, point7, point8
        """
        return (
            (0, 0),
            (0, (self.macro_block_matrix_width - 1) * macro_block_size),
            ((self.macro_block_matrix_height - 1) * macro_block_size, 0),
            ((self.macro_block_matrix_height - 1) * macro_block_size,
             (self.macro_block_matrix_width - 1) * macro_block_size),
            self.macro_block_mappings[0][2],
            self.macro_block_mappings[self.macro_block_matrix_width - 1][2],
            self.macro_block_mappings[(self.macro_block_matrix_height - 1) * self.macro_block_matrix_width][2],
            self.macro_block_mappings[(
                                                  self.macro_block_matrix_height - 1) * self.macro_block_matrix_width + self.macro_block_matrix_width - 1][
                2],
        )

    def bfs_background_marco_block_h_w(self, start_point: (int, int)) -> (int, int):
        """

        :param start_point:
        :return: nearest_point which is not background
        """
        visited = []
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        q = [start_point]
        # print(f"start point:{start_point}",end=":")
        visited.append(start_point)
        # radis=0
        while q:
            layer_size = len(q)
            # print(f"r={radis}",end=',')
            for i in range(layer_size):
                p = q.pop(0)
                visited.append(p)
                # print(p)
                if self.macro_block_labels_allocated_as_blocks[p[0]][p[1]] == 0:
                    # print("got it")
                    # print(f"end point:{p}")
                    return p
                for dir in dirs:

                    neigh = (p[0] + dir[0], p[1] + dir[1])
                    if neigh not in visited and 0 <= neigh[0] < self.macro_block_matrix_height and 0 <= neigh[
                        1] < self.macro_block_matrix_width:
                        # print(f"{p}+{dir}->{neigh}")
                        q.append(neigh)
            # radis+=1

        return start_point

    def get_eight_point_2(self):
        """

        point: (y,x)
        :return: src point: point1, point2, point3, point4
                dst point: point5, point6, point7, point8
        """
        l_t_bg_block = self.bfs_background_marco_block_h_w((0, 0))
        r_t_bg_block = self.bfs_background_marco_block_h_w((0, self.macro_block_matrix_width - 1))
        l_b_bg_block = self.bfs_background_marco_block_h_w(((self.macro_block_matrix_height - 1), 0))
        r_b_bg_block = self.bfs_background_marco_block_h_w(((self.macro_block_matrix_height - 1),
                                                            (self.macro_block_matrix_width - 1)))

        return (
            (l_t_bg_block[0] * macro_block_size, l_t_bg_block[1] * macro_block_size),
            (r_t_bg_block[0] * macro_block_size, r_t_bg_block[1] * macro_block_size),
            (l_b_bg_block[0] * macro_block_size, l_b_bg_block[1] * macro_block_size),
            (r_b_bg_block[0] * macro_block_size, r_b_bg_block[1] * macro_block_size),
            self.macro_block_mappings[l_t_bg_block[0] * self.macro_block_matrix_width + l_t_bg_block[1]][2],
            self.macro_block_mappings[r_t_bg_block[0] * self.macro_block_matrix_width + r_t_bg_block[1]][2],
            self.macro_block_mappings[l_b_bg_block[0] * self.macro_block_matrix_width + l_b_bg_block[1]][2],
            self.macro_block_mappings[r_b_bg_block[0] * self.macro_block_matrix_width + r_b_bg_block[1]][2]
        )

# def read_rgb_file(filepath):
#     res = np.empty((width, height, 3), np.uint8)
#     with open(filepath, 'rb') as f:
#         for w in range(width):
#             for h in range(height):
#                 byte = f.read(3)
#                 r = byte[0]
#                 g = byte[1]
#                 b = byte[2]
#                 res[w, h, 0] = r
#                 res[w, h, 1] = g
#                 res[w, h, 2] = b
#     return res.reshape((height, width, 3))


# if __name__ == '__main__':
#     folder = Path(folder_path)
#     images = []
#     count = 0
#
#     for filepath in folder.iterdir():
#         if count % 10 == 0:
#             print(count)
#         count += 1
#         rgb_array = read_rgb_file(filepath)
#         img = RGBImage(cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))
#         img.display(1)
