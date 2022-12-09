import cv2
import numpy as np


block_size = 16


class RGBImage:

    def __init__(self, arr: np.array, block_arr):
        self.bgr_arr = arr
        self.block_arr = block_arr
        shape = self.block_arr.shape
        self.block_height = shape[0]
        self.block_width = shape[1]

    def getitem(self, h, w):
        return self.bgr_arr[h][w]

    def display(self, wait_time):

        arr = np.copy(self.bgr_arr)
        for i in range(self.block_height):
            for j in range(self.block_width):
                y1, x1, y2, x2 = i*block_size, j*block_size, (i+1)*block_size, (j+1)*block_size
                label = self.block_arr[i, j]
                if label:
                    bgr = (0, 255, 0)
                    cv2.rectangle(arr, (x1, y1), (x2, y2), bgr, 1)
                    cv2.putText(arr, str(int(label)), (x1, y2), cv2.FONT_HERSHEY_PLAIN, .7, bgr)

        cv2.imshow('image', arr)
        cv2.waitKey(wait_time)
        if wait_time == 0:
            cv2.destroyAllWindows()

    def get_background(self):
        """

        :return: background array foreground replace by rgb(0,0,0)
        """
        arr = np.copy(self.bgr_arr)
        for i in range(self.block_height):
            for j in range(self.block_width):
                y1, x1, y2, x2 = i*block_size, j*block_size, (i+1)*block_size, (j+1)*block_size
                label = self.block_arr[i, j]
                if label:
                    bgr = (0, 0, 0)
                    cv2.fillPoly(arr, [np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])], bgr)

        return arr

    def get_foreground(self):
        arr = np.copy(self.bgr_arr)
        for i in range(self.block_height):
            for j in range(self.block_width):
                y1, x1, y2, x2 = i * block_size, j * block_size, (i + 1) * block_size, (j + 1) * block_size
                label = self.block_arr[i, j]
                if not label:
                    bgr = (0, 0, 0)
                    cv2.fillPoly(arr, [np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])], bgr)

        return arr

    def get_macro_block_label(self) -> np.array:
        """

        block_label[i,j] = label of (i,j)
        label = 0 if background, label != 0 if foreground
        :return: block_label: numpy array of shape (height/block_size, width/block_size)
        """
        return self.block_arr

    def get_macro_block(self, i, j) -> np.array:
        """

        :param i: index in height
        :param j: index in width
        :return: numpy array of block_size * block_size
        """
        return self.bgr_arr[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size, :]

    def get_original_bgr_array(self) -> np.array:
        """

        :return:
        """
        return np.copy(self.bgr_arr)

    def get_motion_vector(self, label) -> '(dy, dx)':
        """
        Motion vector based on previous frame
        :param label: label retrieve by get_macro_block_label
        :return: vertical motion in pixels, horizontal motion in pixels
        """
        pass  # todo


    def get_eight_point(self):
        """

        point: (y,x)
        :return: src point: point1, point2, point3, point4
                dst point: point5, point6, point7, point8
        """
        pass  # todo


def read_rgb_file(filepath, width, height):
    res = np.empty((width, height, 3), np.uint8)
    with open(filepath, 'rb') as f:
        for w in range(width):
            for h in range(height):
                byte = f.read(3)
                r = byte[0]
                g = byte[1]
                b = byte[2]
                res[w, h, 0] = r
                res[w, h, 1] = g
                res[w, h, 2] = b

    return res.reshape((height, width, 3))


if __name__ == '__main__':
    pass

