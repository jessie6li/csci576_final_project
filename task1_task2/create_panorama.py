# -*- coding: utf-8 -*-
"""Untitled5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/124-_5sTFyvP3DsEn8p4lJSf9lPK8ql6h
"""

from google.colab.patches import cv2_imshow
import math
import cv2
import numpy as np
import torch
from imageDisplay2 import RGBImage, block_size

file_path = 'SAL.mp4'


class PersonDetection:

    def __init__(self, video_path):
        self.cords = []
        self.labels = []
        self.frames = self._load_frames(video_path)
        self.model = self._load_model()
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using', self.device)

    def _load_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def _load_frames(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        return frames

    def resize(self, height, width):
        for i in range(len(self.frames)):
            frame = self.frames[i]
            resized = cv2.resize(frame, (width,height), interpolation=cv2.INTER_AREA)
            self.frames[i] = resized


    def calc_frame(self, frame):
        self.model.to(self.device)
        res = self.model([frame])
        label = res.xyxyn[0][:, -1]
        cord = res.xyxyn[0][:, :-1]
        return label, cord

    def get_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, label, cord, frame):
        n = len(label)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        new_frame = np.copy(frame)
        for i in range(n):
            r = cord[i]
            if label[i] == 0 and r[4] > .3:
                x1, y1, x2, y2 = int(r[0] * x_shape), int(r[1] * y_shape), int(r[2] * x_shape), int(r[3] * y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(new_frame, (x1, y1), (x2, y2), bgr, 1)
                # cv2.putText(frame, self.get_label(label[i]), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, bgr)
        return new_frame

    def run(self):

        n = len(self.frames)
        for i in range(n):
            if i % 50 == 0:
                print(f'process {i}/{n}')
            frame = self.frames[i]
            label, cord = self.calc_frame(frame)
            self.labels.append(label)
            self.cords.append(cord)
            # new_frame = self.plot_boxes(label, cord, frame)
            # self.frames[i] = new_frame

    def display(self, frame_num):
        cv2.imshow("img", self.frames[frame_num])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_rgbimages(self):
        """

        :return: list of RGBImage Object
        """
        imgs = []
        height, width, _ = self.frames[0].shape
        block_height, block_width = math.ceil(height / block_size), math.ceil(width / block_size)

        for i in range(len(self.frames)):
            label, cord = self.labels[i], self.cords[i]
            block_arr = np.zeros((block_height, block_width))
            label_num = 1
            for j in range(len(label)):
                r = cord[j]
                if label[j] == 0 and r[4] > .3:
                    x1, y1, x2, y2 = int(r[0] * block_width), int(r[1] * block_height), \
                                     int(r[2] * block_width), int(r[3] * block_height)
                    for y in range(y1, y2):
                        for x in range(x1, x2):
                            block_arr[y, x] = label_num
                    label_num += 1
            imgs.append(RGBImage(self.frames[i], block_arr))

        return imgs


def get_pano(imgs):
    bgs = []
    for i in range(len(imgs)):
        if i%50 == 0:
            bgs.append(imgs[i].get_original_bgr_array())
    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    status, pano = stitcher.stitch(bgs)

    if status != cv2.Stitcher_OK:
        print("Can't stitch images, error code = %d" % status)

    cv2.imshow('img', pano)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    pd = PersonDetection(file_path)
    pd.resize(270, 490)
    pd.run()
    original_imgs = pd.get_rgbimages()
    #for i in imgs:
        #i.display(300)
    
    bg = original_imgs[0].get_background()
    cv2_imshow(bg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

bg = original_imgs[100].get_background()
cv2_imshow(bg)

import math
import sys
import time

import numpy as np
import cv2
from imageDisplay_liu import RGBImage

np.set_printoptions(threshold=sys.maxsize)


def read_video_to_frames(path):
    res = []
    cap = cv2.VideoCapture(path)
    currentframe = 0
    while True:
        ret, frame = cap.read()
        # print(frame)
        if ret:
            res.append(frame)
            currentframe += 1
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    return res


video_path = "SAL.mp4"
print(f"read video: {video_path}")
frames = read_video_to_frames(video_path)
frame_size_h = len(frames[0])
frame_size_w = len(frames[0][0])
print(len(frames))
eight_points_list = []
for i in range(1, len(frames)):
    img = RGBImage(frames[i], frames[i - 1], new_size_h_w=(270,490))
    eight_points_list.append(img.get_eight_point_2())

import scipy.stats
def create_panorama_recursive(canvas, pre_H, imgs, eight_points_list, frame):
    if frame == 436:
        return canvas
    else:
        images = imgs
        eight_points_l = eight_points_list
        curr_img = images[frame].get_background()
        eight_points = eight_points_l[frame]
        curr_pts = np.float32(((0, 0), (0, 464), (240, 0), (240, 464)))
        mode_mvs = scipy.stats.mode(np.array(eight_points[4:8])).mode[0]
        curr_mvs = np.float32([mode_mvs, mode_mvs, mode_mvs, mode_mvs])
        #curr_pts = np.float32(((0, 0), (0, 464), (240, 0), (240, 464)))
        #curr_mvs = np.float32(((0,-1), (0,-1), (0,-1), (0,-1)))
        prev_pts = np.subtract(curr_pts, curr_mvs)
        new_prev_pts = []
        for i in range(len(prev_pts)):
            prev_pt = np.append(prev_pts[i], 1)
            product = np.dot(pre_H, [prev_pt[0], prev_pt[1], 1])
            new = product
            #new = product / product[2]
            new_prev_pts.append([math.floor(new[0]), math.floor(new[1])])
        curr_H = cv2.getPerspectiveTransform(curr_pts, np.float32(new_prev_pts))
        max_h = canvas.shape[0] - 1
        max_w = canvas.shape[1] - 1
        h_offset = 0
        w_offset = 0
        for h in range(curr_img.shape[0]):
            for w in range(curr_img.shape[1]):
                product = np.dot(curr_H, [h, w, 1])
                new = product
                #new = product / product[2]
                new_h = math.floor(new[0])
                new_w = math.floor(new[1])
                new_index_h = new_h + h_offset
                new_index_w = new_w + w_offset
                if new_index_h > max_h:
                    canvas = np.pad(canvas, ((0, new_index_h - max_h), (0, 0), (0, 0)), mode='constant', constant_values=0)
                    max_h = new_index_h
                    #print("in 1")
                elif new_index_h < 0:
                    canvas = np.pad(canvas, ((0 - new_index_h, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
                    h_offset += -new_index_h
                    new_index_h = 0
                    #print("in 2")
                if new_index_w > max_w:
                    canvas = np.pad(canvas, ((0, 0), (0, new_index_w - max_w), (0, 0)), mode='constant', constant_values=0)
                    max_w = new_index_w
                elif new_index_w < 0:
                    canvas = np.pad(canvas, ((0, 0), (0 - new_index_w, 0), (0, 0)), mode='constant', constant_values=0)
                    w_offset += -new_index_w
                    new_index_w = 0
                if (np.array_equal(canvas[new_index_h][new_index_w], [0, 0, 0])):
                    canvas[new_index_h][new_index_w] = curr_img[h][w]
    
        return create_panorama_recursive(canvas, curr_H, images, eight_points_l, frame+1)

H = [[1,0,0], [0,1,0], [0,0,1]]
canvas = original_imgs[0].get_background()
res = create_panorama_recursive(canvas, H, original_imgs[1:len(original_imgs)], eight_points_list, 0)
print(res.shape)
cv2_imshow(res)

cv2.imwrite("panorama_SAL.jpg", res)

def create_panorama_recursive2(old_canvas, pre_H, imgs, eight_points_list, frame):
    if frame == 100:
        return old_canvas
    else:
        images = imgs
        eight_points_l = eight_points_list 
        curr_img = images[frame].get_background()
        eight_points = eight_points_list[frame]
        curr_pts = np.float32(eight_points[0:4])
        curr_mvs = np.float32(eight_points[4:8])
        prev_pts = np.subtract(curr_pts, curr_mvs)
        new_prev_pts = []
        for i in range(len(prev_pts)):
            prev_pt = np.append(prev_pts[i], 1)
            product = np.dot(pre_H, [prev_pt[0], prev_pt[1], 1])
            new = product
            new_prev_pts.append([math.floor(new[0]), math.floor(new[1])])
        curr_H = cv2.getPerspectiveTransform(curr_pts, np.float32(new_prev_pts))
        max_h = old_canvas.shape[0] - 1
        min_h = 0
        max_w = old_canvas.shape[1] - 1
        min_w = 0
        for h in range(curr_img.shape[0]):
            for w in range(curr_img.shape[1]):
                product = np.dot(curr_H, [h,w,1])
                new = product
                new_h = math.floor(new[0])
                new_w = math.floor(new[1])
                # max_h = new_h if new_h > max_h else max_h
                # min_h = new_h if new_h < min_h else min_h
                # max_w = new_w if new_w > max_w else max_w
                # min_w = new_w if new_w < min_w else min_w
                if new_h > max_h:
                  max_h = new_h
                if new_h < min_h:
                  min_h = new_h
                if new_w > max_w:
                  max_w = new_w
                if new_w < min_w:
                  min_w = new_w  
        new_canvas_height = math.ceil(max_h - min_h)+1
        # print("h")
        # print(max_h)
        # print(min_h)
        # print("w")
        # print(max_w)
        # print(min_w)
        # print("new")
        # print(new_h)
        # print(new_w)
        new_canvas_width = math.ceil(max_w - min_w)+1
        new_canvas = np.zeros((new_canvas_height, new_canvas_width, 3), np.uint8)
        # print(new_canvas.shape)
        # print(0-min_h)
        # print(canvas.shape[0]-min_h)
        # print(0-min_w)
        # print(canvas.shape[1]-min_w)
        new_canvas[0-min_h:canvas.shape[0]-min_h, 0-min_w:canvas.shape[1]-min_w] = canvas
        for h in range(curr_img.shape[0]):
            for w in range(curr_img.shape[1]):
                product = np.dot(curr_H, [h, w, 1])
                new = product
                new_h = math.floor(new[0])
                new_w = math.floor(new[1])
                #print(new_h)
                #print("new2")
                # print(new_h)
                #print(new_w)
                if (np.array_equal(new_canvas[new_h][new_w], [0, 0, 0])):
                    new_canvas[new_h][new_w] = curr_img[h][w]
    
        return create_panorama_recursive2(new_canvas, curr_H, images, eight_points_list, frame+1)

H = [[1,0,0], [0,1,0], [0,0,1]]
canvas = original_imgs[0].get_background()
res2 = create_panorama_recursive2(canvas, H, original_imgs[1:len(original_imgs)+1], eight_points_list, 0)
print(res2.shape)
cv2_imshow(res2)

canvas = original_imgs[0].get_background()
cv2_imshow(canvas)