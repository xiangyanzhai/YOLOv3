# !/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf

base_anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
base_anchors = tf.constant(base_anchors, dtype=tf.float32)
base_anchors = tf.reshape(base_anchors, (3, 3, 2))
base_anchors = base_anchors[::-1]
anchors = tf.reshape(base_anchors, (-1, 2))
anchors_shift = tf.concat([-anchors / 2, anchors / 2], axis=-1)


def cal_IOU(bboxes, anchors):
    # bboxes m*b*4
    # anchors a*4

    hw = bboxes[..., 2:4] - bboxes[..., :2]
    areas1 = tf.reduce_prod(hw, axis=-1)  # m*b

    hw = anchors[:, 2:4] - anchors[:, :2]
    areas2 = tf.reduce_prod(hw, axis=-1)  # a

    yx1 = tf.maximum(bboxes[..., None, :2], anchors[:, :2])  # m*b*a*2
    yx2 = tf.minimum(bboxes[..., None, 2:4], anchors[:, 2:4])  # m*b*a*2

    hw = yx2 - yx1
    hw = tf.maximum(hw, 0)
    areas_i = tf.reduce_prod(hw, axis=-1)  # m*b*a
    iou = areas_i / (areas1[:, None] + areas2 - areas_i)

    # m*b*a
    return iou


def bbox2loc(anchor, bbox):
    c_hw = anchor[..., 2:4] - anchor[..., 0:2]
    c_yx = anchor[..., :2] + c_hw / 2
    hw = bbox[..., 2:4] - bbox[..., 0:2]
    yx = bbox[..., :2] + hw / 2
    t_yx = (yx - c_yx) / c_hw
    t_hw = tf.log(hw / c_hw)
    return tf.concat([t_yx, t_hw], axis=1)


def assign_target(c_target, c_scale, t_j_i_a):
    t = np.split(t_j_i_a, 3, axis=-1)
    c_target[t] = 1
    c_scale[t] = 1

    return c_target, c_scale


def yolov3_loss(out, pre_bboxes, bboxes, map_H, img_size, num_cls, ignore_thresh, ):
    wh = bboxes[..., 2:4] - bboxes[..., :2]
    xy = bboxes[..., :2] + wh / 2
    bboxes_shift = bboxes[..., :4] - tf.concat([xy, xy], axis=-1)
    iou_shift = cal_IOU(bboxes_shift, anchors_shift)

    a = tf.argmax(iou_shift, axis=-1)
    c_iou = tf.reduce_max(cal_IOU(pre_bboxes, bboxes), axis=-1)

    loss = 0
    C = 0

    for i in range(3):
        t_H = map_H * 2 ** i
        c = t_H * t_H * 3
        t_out = tf.reshape(out[C:C + c], (t_H, t_H, 3, num_cls + 5))
        t_c_iou = tf.reshape(c_iou[C:C + c], (t_H, t_H, 3))
        C += c

        inds = (a >= (i * 3)) & (a < ((i + 1) * 3))
        t_xy = tf.boolean_mask(xy, inds)
        t_ji = tf.to_int64(tf.floor(t_xy / 32 * 2 ** i)[..., ::-1])
        t_xy = tf.mod(t_xy, 32 / (2 ** i)) / (32 / (2 ** i))
        t_wh = tf.boolean_mask(wh, inds)
        scale_coord = 2 - tf.reduce_prod(t_wh, axis=-1, keepdims=True) / tf.to_float(img_size ** 2)
        t_a = tf.boolean_mask(a, inds) - i * 3
        t_anchor = tf.gather(base_anchors[i], t_a)
        t_wh = tf.log(t_wh / t_anchor)

        t_j_i_a = tf.concat((t_ji, tf.reshape(t_a, (-1, 1))), axis=-1)
        pre = tf.gather_nd(t_out, t_j_i_a)
        pre_xy = tf.sigmoid(pre[..., :2])
        pre_wh = pre[..., 2:4]

        pre_cls = pre[..., 5:]

        loss1 = tf.reduce_sum(scale_coord * ((pre_xy - t_xy) ** 2 + (pre_wh - t_wh) ** 2)) / 2 \
                + tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=pre_cls,
                                                                        labels=tf.one_hot(tf.to_int32(
                                                                            tf.boolean_mask(bboxes[..., -1], inds)),
                                                                            num_cls)))

        c_scale = tf.to_float(t_c_iou <= ignore_thresh)
        if True:
            c_target = tf.zeros(tf.shape(c_scale), dtype=tf.float32)
            c_target, c_scale = tf.py_func(assign_target, [c_target, c_scale, t_j_i_a], [tf.float32, tf.float32])
        else:
            c_target = tf.scatter_nd(tf.to_int32(t_j_i_a), tf.ones(tf.shape(t_j_i_a)[0]), tf.shape(c_scale))
            c_scale += c_target
            c_scale = tf.clip_by_value(c_scale, 0, 1)

        loss2 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=t_out[..., 4], labels=c_target) * c_scale)

        loss += tf.cond(tf.equal(tf.shape(t_xy)[0], 0), lambda: loss2, lambda: loss1 + loss2)
    return loss


if __name__ == "__main__":
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    x = tf.range(10)
    b = x > 10
    c = tf.boolean_mask(x, b)
    d = c / 2
    with tf.Session() as sess:
        print(sess.run(d))
    # t=tf.constant(list(range(9)))
    # t=t>5
    # t=tf.boolean_mask(anchors_shift,t)
    # with tf.Session() as sess:
    #     t = sess.run(base_anchors)
    #     print(t)
    #     t = sess.run(anchors)
    #     print(t)
    #     t = sess.run(anchors_shift)
    #     print(t)
    pass
