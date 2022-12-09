import math
import cv2
import numpy as np
import torch
from imageDisplay2 import RGBImage, block_size

file_path = 'video/test2.mp4'


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

    def resize(self, width, height):
        for i in range(len(self.frames)):
            frame = self.frames[i]
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            self.frames[i] = resized
        print(self.frames[0].shape)

    def calc_frame(self, frame):
        self.model.to(self.device)
        res = self.model([frame])
        label = res.xyxyn[0][:, -1]
        cord = res.xyxyn[0][:, :-1]
        return label, cord

    def get_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, frame):
        x_shape, y_shape = frame.shape[1] // block_size, frame.shape[0] // block_size
        new_frame = np.copy(frame)
        for i in range(x_shape - 1):
            for j in range(y_shape - 1):
                x1, y1, x2, y2 = int(i * block_size), int(j * block_size), int((i + 1) * block_size), int((j+1) * block_size)
                bgr = (0, 255, 0)
                cv2.rectangle(new_frame, (x1, y1), (x2, y2), bgr, 1)
        return new_frame

    def plot_label_boxes(self, label, cord, frame):
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
            # new_frame = self.plot_boxes(frame)
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
                                     min(int(r[2] * block_width)+1, block_width), \
                                     min(int(r[3] * block_height)+1, block_height)
                    for y in range(y1, y2):
                        for x in range(x1, x2):
                            block_arr[y, x] = label_num
                    label_num += 1
            imgs.append(RGBImage(self.frames[i], block_arr))

        return imgs


def get_pano(imgs):
    bgs = []
    for i in range(len(imgs)):
        if i % 40 == 0:
            bgs.append(imgs[i].get_background())
    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    status, pano = stitcher.stitch(bgs)

    if status != cv2.Stitcher_OK:
        print("Can't stitch images, error code = %d" % status)

    cv2.imshow('img', pano)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    pd = PersonDetection(file_path)
    pd.resize(480, 272)
    pd.run()
    imgs = pd.get_rgbimages()
    for i, v in enumerate(imgs):
        v.display(100)
    # get_pano(imgs)
    # bg = imgs[0].get_background()
    # cv2.imshow('img', bg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
