import cv2
import time
from skimage import io
import math
import os
import numpy as np
import tensorflow as tf
import sys
import locality_aware_nms as nms_locality
import lanms
from io import StringIO


tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', '/goshposh/app/east_icdar2015_resnet_v1_50_rbox/', '')

import model
from icdar import restore_rectangle

FLAGS = tf.app.flags.FLAGS


def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resize_h = max(32, resize_h)
    resize_w = max(32, resize_w)
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    # boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def main(argv=None):
    og = sys.stdout
    sys.stdout = StringIO()
    #str(sys.argv[1])
    image_url = "http://di2ponv0v5otw.cloudfront.net/posts/2019/02/18/5c6b5643bb76153bac47d33f/m_5c6b565af63eeaf2a2ffaf19.jpg"
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        f_score, f_geometry = model.model(input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        max_width = 580
        max_height = 580

        matrix_size = 6

        w = max_width / matrix_size
        h = max_height / matrix_size

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            # print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im = io.imread(image_url)[:, :, ::-1]
            # start_time = time.time()
            im_resized, (ratio_h, ratio_w) = resize_image(im)

            timer = {'net': 0, 'restore': 0, 'nms': 0}
            start = time.time()
            score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
            timer['net'] = time.time() - start

            boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)

            if boxes is not None:
                boxes = boxes[:, :8].reshape((-1, 4, 2))
                boxes[:, :, 0] /= ratio_w
                boxes[:, :, 1] /= ratio_h

            # duration = time.time() - start_time
            # print('[timing] {}'.format(duration))

            quadrant_list = []
            # save to file
            if boxes is not None:
                for box in boxes:
                    # to avoid submitting errors
                    box = sort_poly(box.astype(np.int32))
                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                        continue
                    for index in range(len(box)):
                        point_w = min(math.ceil(box[index][0]), max_width)
                        point_h = min(math.ceil(box[index][1]), max_height)
                        quadrant_list.append(math.ceil(point_w / w) + 6 * (max(math.ceil(point_h / h) - 1, 0)))
            sys.stdout = og
            print(set(quadrant_list))

if __name__ == '__main__':
    tf.app.run()