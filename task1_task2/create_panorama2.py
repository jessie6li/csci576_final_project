import scipy.stats
import numpy as np
import cv2,math
from foo import eight_point1, eight_point2
from predYOLO import PersonDetection

def create_panorama_recursive_key_frame(canvas, pre_H, imgs, eight_points_list, frame,
                                        acc_h_offset, acc_w_offset, key_frame_count):
    if frame == len(imgs):
        return canvas
    else:
        if frame % 10 == 0:
            print(f'{frame}/{len(imgs)}')
        images = imgs
        eight_points_l = eight_points_list
        if frame%key_frame_count == 0:
            curr_img = images[frame].get_foreground()
        else:
            curr_img = images[frame].get_background()
        eight_points = eight_points_l[frame]
        curr_pts = np.float32(((0, 0), (0, 464), (240, 0), (240, 464)))
        # mode_mvs = scipy.stats.mode(np.array(eight_points[4:8])).mode[0]
        mode_mvs = (0, -1)
        curr_mvs = np.float32([mode_mvs, mode_mvs, mode_mvs, mode_mvs])
        # curr_pts = np.float32(eight_points[0:4])
        # curr_mvs = np.float32(eight_points[4:8])
        # curr_pts = np.float32(((0, 0), (0, 464), (240, 0), (240, 464)))
        # curr_mvs = np.float32(((0,-1), (0,-1), (0,-1), (0,-1)))
        prev_pts = np.subtract(curr_pts, curr_mvs)

        new_prev_pts = []
        for i in range(len(prev_pts)):
            prev_pt = np.append(prev_pts[i], 1)
            product = np.dot(pre_H, [prev_pt[0], prev_pt[1], 1])
            new = product
            # new = product / product[2]
            new_prev_pts.append([math.floor(new[0])+acc_h_offset, math.floor(new[1])+acc_w_offset])

        curr_H = cv2.getPerspectiveTransform(curr_pts, np.float32(new_prev_pts))
        max_h = canvas.shape[0] - 1
        max_w = canvas.shape[1] - 1
        h_offset = 0
        w_offset = 0
        black = np.array([0,0,0])

        for h in range(curr_img.shape[0]):
            for w in range(curr_img.shape[1]):
                product = np.dot(curr_H, [h, w, 1])
                new = product
                # new = product / product[2]
                new_h = math.floor(new[0])
                new_w = math.floor(new[1])
                new_index_h = new_h + h_offset
                new_index_w = new_w + w_offset
                if new_index_h > max_h:
                    canvas = np.pad(canvas, ((0, new_index_h - max_h), (0, 0), (0, 0)), mode='constant',
                                    constant_values=0)
                    max_h = new_index_h
                    # print("in 1")
                elif new_index_h < 0:
                    canvas = np.pad(canvas, ((0 - new_index_h, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
                    h_offset += -new_index_h
                    new_index_h = 0
                    # print("in 2")
                if new_index_w > max_w:
                    canvas = np.pad(canvas, ((0, 0), (0, new_index_w - max_w), (0, 0)), mode='constant',
                                    constant_values=0)
                    max_w = new_index_w
                elif new_index_w < 0:
                    canvas = np.pad(canvas, ((0, 0), (0 - new_index_w, 0), (0, 0)), mode='constant', constant_values=0)
                    w_offset += -new_index_w
                    new_index_w = 0

                if frame % key_frame_count == 0 and not np.array_equal(curr_img[h][w], black):
                    canvas[new_index_h][new_index_w] = curr_img[h][w]
                elif np.array_equal(canvas[new_index_h][new_index_w], black):
                    canvas[new_index_h][new_index_w] = curr_img[h][w]
        cv2.imwrite(f'intermediate/{frame}.jpg',canvas)
        return create_panorama_recursive_key_frame(canvas, curr_H, images, eight_points_l, frame + 1, acc_h_offset + h_offset,
                                         acc_w_offset + w_offset,key_frame_count)


def create_panorama_recursive(canvas, pre_H, imgs, eight_points_list, frame, acc_h_offset, acc_w_offset):
    if frame == len(imgs):
        return canvas
    else:
        if frame % 10 == 0:
            print(f'{frame}/{len(imgs)}')
        images = imgs
        eight_points_l = eight_points_list
        curr_img = images[frame].get_background()
        eight_points = eight_points_l[frame]
        curr_pts = np.float32(((0, 0), (0, 464), (240, 0), (240, 464)))
        mode_mvs = scipy.stats.mode(np.array(eight_points[4:8])).mode[0]

        curr_mvs = np.float32([mode_mvs, mode_mvs, mode_mvs, mode_mvs])
        # curr_pts = np.float32(eight_points[0:4])
        # curr_mvs = np.float32(eight_points[4:8])
        # curr_pts = np.float32(((0, 0), (0, 464), (240, 0), (240, 464)))
        # curr_mvs = np.float32(((0,-1), (0,-1), (0,-1), (0,-1)))
        prev_pts = np.subtract(curr_pts, curr_mvs)

        new_prev_pts = []
        for i in range(len(prev_pts)):
            prev_pt = np.append(prev_pts[i], 1)
            product = np.dot(pre_H, [prev_pt[0], prev_pt[1], 1])
            new = product
            # new = product / product[2]
            new_prev_pts.append([math.floor(new[0])+acc_h_offset, math.floor(new[1])+acc_w_offset])
        print(frame)
        print('cur:', curr_mvs, sep='\n')
        print('prev', new_prev_pts, sep='\n')
        curr_H = cv2.getPerspectiveTransform(curr_pts, np.float32(new_prev_pts))
        max_h = canvas.shape[0] - 1
        max_w = canvas.shape[1] - 1
        h_offset = 0
        w_offset = 0
        black = np.array([0,0,0])
        for h in range(curr_img.shape[0]):
            for w in range(curr_img.shape[1]):
                product = np.dot(curr_H, [h, w, 1])
                new = product
                # new = product / product[2]
                new_h = math.floor(new[0])
                new_w = math.floor(new[1])
                new_index_h = new_h + h_offset
                new_index_w = new_w + w_offset
                if new_index_h > max_h:
                    canvas = np.pad(canvas, ((0, new_index_h - max_h), (0, 0), (0, 0)), mode='constant',
                                    constant_values=0)
                    max_h = new_index_h
                    # print("in 1")
                elif new_index_h < 0:
                    canvas = np.pad(canvas, ((0 - new_index_h, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
                    h_offset += -new_index_h
                    new_index_h = 0
                    # print("in 2")
                if new_index_w > max_w:
                    canvas = np.pad(canvas, ((0, 0), (0, new_index_w - max_w), (0, 0)), mode='constant',
                                    constant_values=0)
                    max_w = new_index_w
                elif new_index_w < 0:
                    canvas = np.pad(canvas, ((0, 0), (0 - new_index_w, 0), (0, 0)), mode='constant', constant_values=0)
                    w_offset += -new_index_w
                    new_index_w = 0
                if np.array_equal(canvas[new_index_h][new_index_w], black):
                    canvas[new_index_h][new_index_w] = curr_img[h][w]
        cv2.imwrite(f'intermediate/{frame}.jpg',canvas)
        return create_panorama_recursive(canvas, curr_H, images, eight_points_l, frame + 1, acc_h_offset + h_offset,
                                         acc_w_offset + w_offset)




if __name__ == '__main__':
    pd = PersonDetection('video/test2.mp4')
    pd.resize(490, 270)
    pd.run()
    original_imgs = pd.get_rgbimages()
    eight_points_list = eight_point2
    H = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    canvas = original_imgs[0].get_background()
    res = create_panorama_recursive(canvas, H, original_imgs[1:len(original_imgs)], eight_points_list, 0, 0, 0)
    print(res.shape)
    cv2.imshow('img', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
