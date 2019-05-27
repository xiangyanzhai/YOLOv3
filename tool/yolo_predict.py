# !/usr/bin/python
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np


def py_inds(score, inds):
    score[inds] = score[inds] * -1
    inds = score > 0
    score[inds] = 0
    score = score * -1

    return score


def fn_map(x):
    bboxes = x[0]
    score = x[1]

    inds = tf.image.non_max_suppression(bboxes, score, tf.shape(bboxes)[0], iou_threshold=iou_thresh)
    score = tf.py_func(py_inds, [score, inds], tf.float32)

    return score


# def fn_map(x):
#     bboxes = x[0]
#     score = x[1]
#     m = tf.shape(score)
#     inds = tf.image.non_max_suppression(bboxes, score, tf.shape(bboxes)[0], iou_threshold=iou_thresh)
#     inds = inds[..., None]
#     mask = tf.scatter_nd(inds, tf.ones(tf.shape(inds)[0]), m)
#     score = score * (tf.to_float(mask))
#     return score


def predict(net, size, num_cls, iou_thresh_=0.45, c_thresh=1e-3):
    global iou_thresh
    iou_thresh = iou_thresh_
    inds = net[..., 4] > c_thresh
    net = tf.boolean_mask(net, inds)

    bboxes = net[..., :4]

    # m*4 m*1 m*20
    bboxes = tf.clip_by_value(bboxes, 0, size)
    mbboxes = tf.expand_dims(bboxes, axis=0)
    mbboxes = tf.tile(mbboxes, [num_cls, 1, 1])
    score = tf.transpose(net[..., 5:]) * net[..., 4]

    score = tf.map_fn(fn_map, [mbboxes, score], back_prop=False, dtype=tf.float32)

    score = tf.transpose(score)
    cls = tf.argmax(score, axis=1)
    score = tf.reduce_max(score, axis=1)
    cls = tf.to_float(cls)
    score = tf.reshape(score, (-1, 1))
    cls = tf.reshape(cls, (-1, 1))

    pre = tf.concat([bboxes, score, cls], axis=1)
    # pre = tf.boolean_mask(pre, pre[:, -2] > c_thresh)
    _, top_k = tf.nn.top_k(pre[:, -2], tf.shape(pre)[0])
    pre = tf.gather(pre, top_k)
    return pre


# def predict(bboxes, conf, score, size, num_cls, iou_thresh_=0.45, c_thresh=1e-3):
#     global iou_thresh
#     conf=tf.sigmoid(conf)
#     iou_thresh = iou_thresh_
#     inds = conf > c_thresh
#     bboxes = tf.boolean_mask(bboxes, inds)
#     score = tf.boolean_mask(score, inds)
#     conf = tf.boolean_mask(conf, inds)
#     # m*4 m*1 m*20
#     bboxes = tf.clip_by_value(bboxes, 0, size)
#     mbboxes = tf.expand_dims(bboxes, axis=0)
#     mbboxes = tf.tile(mbboxes, [num_cls, 1, 1])
#     score = tf.transpose(tf.nn.sigmoid(score)) * (conf)
#
#     score = tf.map_fn(fn_map, [mbboxes, score], back_prop=False, dtype=tf.float32)
#
#     score = tf.transpose(score)
#     cls = tf.argmax(score, axis=1)
#     score = tf.reduce_max(score, axis=1)
#     cls = tf.to_float(cls)
#     score = tf.reshape(score, (-1, 1))
#     cls = tf.reshape(cls, (-1, 1))
#
#     pre=tf.concat([bboxes, score, cls], axis=1)
#     pre = tf.boolean_mask(pre, pre[:, -2] > c_thresh)
#     _, top_k = tf.nn.top_k(pre[:, -2], tf.shape(pre)[0])
#     pre = tf.gather(pre, top_k)
#     return pre


if __name__ == "__main__":
    pass
