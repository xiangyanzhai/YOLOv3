# !/usr/bin/python
# -*- coding:utf-8 -*-
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys

sys.path.append('../../')
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import YOLOv3.tool.yolo_utils as yolo_utils
from YOLOv3.tool.config import Config
from YOLOv3.tool.read_Data import readData
from YOLOv3.tool.yolov3_loss import yolov3_loss
from datetime import datetime

base_anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
base_anchors = tf.constant(base_anchors, dtype=tf.float32)
base_anchors = tf.reshape(base_anchors, (3, 3, 2))
base_anchors = base_anchors[::-1]


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = [tf.expand_dims(g, 0) for g, _ in grad_and_vars]
        grads = tf.concat(grads, 0)
        grad = tf.reduce_mean(grads, 0)
        grad_and_var = (grad, grad_and_vars[0][1])
        # [(grad0, var0),(grad1, var1),...]
        average_grads.append(grad_and_var)
    return average_grads


def get_coord(N, stride):
    t = tf.range(int(N / stride))
    x, y = tf.meshgrid(t, t)

    x = x[..., None]
    y = y[..., None]
    coord = tf.concat((x, y, x, y), axis=-1)
    coord = coord[:, :, None, :]
    coord = tf.to_float(coord) * stride
    return coord


def loc2bboxes(net, anchors, coord, stride):
    xy = tf.sigmoid(net[..., :2]) * stride

    wh = tf.exp(net[..., 2:4]) * anchors
    xy1 = xy - wh / 2
    xy2 = xy + wh / 2
    bboxes = tf.concat((xy1, xy2), axis=-1) + coord
    return bboxes


class YOLOv3():

    def __init__(self, config):
        self.config = config
        self.stride = config.stride
        self.blocks = yolo_utils.parse_cfg(config.cfg_file)
        self.coords = []
        for i in range(len(self.stride)):
            self.coords.append(get_coord(config.img_size, self.stride[i]))

    def init(self, sess, Name, init_vars):
        W = yolo_utils.load_weights(self.config.weight_file, self.blocks, Name, init_vars, (80 + 5) * 3, count=5)
        print('***********', len(W), len(init_vars))
        for i in range(len(W)):
            weight = W[i]
            if init_vars[i].get_shape().as_list()[-1] != (self.config.num_cls + 5) * 3:
                print(i, init_vars[i].name, init_vars[i].shape, weight.shape)
                sess.run(init_vars[i].assign(weight))
            else:
                print('*************************')
        print('init finish')
        pass

    def fn_map(self, x):
        C = x[0]
        pre_bboxes = x[1]
        bboxes = x[2][:x[3]]
        loss = yolov3_loss(C, pre_bboxes, bboxes, self.map_H, self.size, self.config.num_cls, self.config.ignore_thresh)
        return loss

    def get_loss(self, C, pre_bboxes, bboxes, nums, ):
        loss = tf.map_fn(self.fn_map, [C, pre_bboxes, bboxes, nums], tf.float32)
        return loss

    def build_net(self, Iter, size):

        self.size = size
        imgs, bboxes, nums = Iter.get_next()

        imgs = tf.image.resize_images(imgs, (size, size))
        scale = tf.to_float(size) / self.config.read_img_size
        m = tf.shape(imgs)[0]
        bboxes = tf.concat([bboxes[..., :4] * scale, bboxes[..., 4:]], axis=-1)
        # param_regularizers = {'gamma': slim.l2_regularizer(self.config.weight_decay)}
        with slim.arg_scope([slim.batch_norm], is_training=True, scale=True, decay=0.9, epsilon=1e-5,
                            ):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(self.config.weight_decay)):
                Name, end_points = yolo_utils.create_net(imgs / 255, self.blocks)
        self.Name = Name
        init_vars = tf.global_variables()[2:]
        for i in range(len(Name)):
            print(i, '\t', Name[i], '\t', end_points[i])

        net13 = end_points[82]
        net26 = end_points[94]
        net52 = end_points[-1]

        del end_points
        # for i in range(len(end_points)-1):
        #     if i not in [82,94]:
        #         del end_points[i]
        map_H = tf.shape(net13)[1]
        self.map_H = map_H
        C = [net13, net26, net52]

        tcoords = []
        pre_bboxes = []
        for i in range(3):
            tH = map_H * 2 ** i
            tcoords.append(self.coords[i][:tH, :tH])
            C[i] = tf.reshape(C[i], (-1, tH, tH, 3, self.config.num_cls + 5))

            pre_bboxes.append(tf.reshape(loc2bboxes(C[i], base_anchors[i], tcoords[i], self.stride[i]), (m, -1, 4)))
        C = list(map(lambda c: tf.reshape(c, (m, -1, self.config.num_cls + 5)), C))
        C = tf.concat(C, axis=1)
        pre_bboxes = tf.concat(pre_bboxes, axis=1)
        loss = self.get_loss(C, pre_bboxes, bboxes, nums)
        return tf.reduce_sum(loss) / tf.to_float(m), init_vars

    def train(self):
        shard_nums = self.config.gpus
        batch_size = self.config.gpus * self.config.batch_size_per_GPU
        print('******************* batch_size', batch_size)
        print('*******************  shard_nums', shard_nums)
        base_lr = self.config.lr
        print('******************* base_lr', base_lr)
        steps = tf.Variable(0.0, name='yolo', trainable=False)
        x = 1
        lr = tf.case({steps < 40000.0 * x: lambda: base_lr,
                      steps < 50000.0 * x: lambda: base_lr / 10},
                     default=lambda: base_lr / 100)

        tower_grads = []
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        var_reuse = False
        Iter_list = []

        c = 0
        size_num = self.config.size_num
        Size = np.arange(size_num, dtype=np.int32) * 32 + 320
        Size = tf.constant(Size)
        size_inds = tf.Variable(0, dtype=tf.int32, trainable=False)
        tt = tf.cond(tf.equal(steps % 10, 0), lambda: tf.random_uniform(shape=(), maxval=size_num, dtype=tf.int32),
                     lambda: size_inds)

        size_inds = size_inds.assign(tt)
        size = Size[size_inds]

        for i in range(shard_nums):
            with tf.device('/gpu:%d' % i):
                print('************ shard_index', c)
                Iter = readData(self.config.files, self.config, batch_size=self.config.batch_size_per_GPU,
                                num_threads=16,
                                shuffle_buffer=1024,
                                num_shards=shard_nums, shard_index=c)
                Iter_list.append(Iter)
                with tf.variable_scope('', reuse=var_reuse):
                    if c == 0:
                        loss, init_vars = self.build_net(Iter, size)

                    else:
                        loss, _ = self.build_net(Iter, size)

                    var_reuse = True

                c += 1

                train_vars = tf.trainable_variables()
                l2_loss = tf.losses.get_regularization_losses()
                l2_re_loss = tf.add_n(l2_loss)
                yolov3_loss = loss + l2_re_loss
                grads_and_vars = opt.compute_gradients(yolov3_loss, train_vars)
                tower_grads.append(grads_and_vars)

        for v in tf.global_variables():
            print(v)

        grads = average_gradients(tower_grads)
        grads = list(zip(*grads))[0]

        grads, norm = tf.clip_by_global_norm(grads, 100.0)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.apply_gradients(zip(grads, train_vars), global_step=steps)
        saver = tf.train.Saver(max_to_keep=200)
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        # file = '/home/zhai/PycharmProjects/Demo35/YOLOv3/train/models/yolov3_voc_1.ckpt-25000'
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            self.init(sess, self.Name, init_vars)

            # saver.restore(sess, file)

            for Iter in Iter_list:
                sess.run(Iter.initializer)

            for i in range(00000, 50200 * x):
                if i % 20 == 0:

                    _, yolov3_loss_, loss_, l2_re_loss_, size_, norm_, lr_ = sess.run(
                        [train_op, yolov3_loss, loss, l2_re_loss, size, norm, lr])

                    print(datetime.now(), 'yolov3_loss:%.4f' % yolov3_loss_, 'loss:%.4f' % loss_,
                          'l2_re_loss:%.4f' % l2_re_loss_, 'norm:%.4f' % norm_, i, size_, lr_)

                else:
                    sess.run(train_op)

                if (i + 1) % 5000 == 0 or ((i + 1) % 1000 == 0 and i < 10000):
                    saver.save(sess, os.path.join('./models/', 'yolov3_voc_1.ckpt'), global_step=i + 1)
            saver.save(sess, os.path.join('./models/', 'yolov3_voc_1.ckpt'), global_step=i + 1)

    pass


if __name__ == "__main__":
    Mean = np.array([123.68, 116.78, 103.94], dtype=np.float32)
    path = '/home/zhai/PycharmProjects/Demo35/data_set_yxyx/'
    files = [path + 'voc_07.tf', path + 'voc_12.tf']
    cfgfile = '/home/zhai/PycharmProjects/Demo35/yolo/cfg/yolov3-voc.cfg'
    weightsfile = '/home/zhai/PycharmProjects/Demo35/yolo/yolo_weights/darknet53.conv.74'

    config = Config(files, cfgfile, weightsfile, gpus=1, batch_size_per_GPU=16, lr=0.001 / 2, img_size=608,
                    read_img_size=416, size_num=8,
                    ignore_thresh=0.5, crop_iou=0.45, keep_ratio=1 / 5., jitter_ratio=[0.3, 0.5, 0.7])
    yolov3 = YOLOv3(config)
    yolov3.train()
    pass
