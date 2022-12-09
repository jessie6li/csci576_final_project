from predYOLO import PersonDetection
import cv2
from create_panorama2 import create_panorama_recursive, create_panorama_recursive_key_frame
from foo import eight_point2

class Generation:
    width = 480
    height = 272

    def __init__(self, filepath):
        pd = PersonDetection(filepath)
        pd.resize(self.width, self.height)
        pd.run()
        self.images = pd.get_rgbimages()

    def output1(self):
        """
        Panorama Generation
        :return:
        """
        H = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        canvas = self.images[0].get_background()
        eight_points_list = eight_point2
        res = create_panorama_recursive(canvas, H, self.images[1:len(self.images)], eight_points_list, 0, 0, 0)
        cv2.imwrite('output/output1.jpg', res)

    def output2(self, mode=0):
        """
        Foreground video Generation
        :return:
        """

        op = []
        for i in self.images:
            op.append(i.get_foreground())
        if mode == 1:
            for i in op:
                cv2.imshow('img', i)
                cv2.waitKey(20)
        else:
            result = cv2.VideoWriter('output/output2.avi',
                                     cv2.VideoWriter_fourcc(*'MJPG'),
                                     30, (self.width, self.height))
            for i in op:
                result.write(i)
            result.release()
            cv2.destroyAllWindows()

    def output3(self, mode=0):
        """
        This is a static image, the same panorama as in section 1 above, but with foreground elements consistently
        and uniformly composited in. Note we are NOT reassessing the quality of the panorama generation but assessing
        the correctness of the blending process to create a motion trail.
        :return:
        """

        H = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        canvas = self.images[0].get_background()
        eight_points_list = eight_point2
        res = create_panorama_recursive_key_frame(canvas, H, self.images[1:len(self.images)], eight_points_list,
                                                  0, 0, 0, 60)
        if mode == 1:
            cv2.imshow('img', res)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            cv2.imwrite('output/output3.jpg', res)

    def output4(self):
        """
        This must be a normal but "new" video, which is a different path through the panorama – a path of your choice
        :return:
        """
        pass

    def output5(self):
        """
        This must look like the original input video with foreground removed. As a guideline, think of the output to
        represent re-filming the same video from the same camera positions at each frame, but with no foreground
        elements present while filming. All motion is due to camera.
         return:
        """
        pass

if __name__ == '__main__':
    g = Generation('video/test2.mp4')
    g.output3()
