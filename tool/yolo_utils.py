# !/usr/bin/python
# -*- coding:utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.slim as slim
import codecs
import numpy as np
import copy

def handle_im_same_wh(im_input,size):
    print(size)
    im = tf.image.decode_jpeg(im_input, channels=3)
    h = tf.shape(im)[0]
    w = tf.shape(im)[1]
    h = tf.to_float(h)
    w = tf.to_float(w)
    ma=tf.reduce_max([h,w])

    s=size/ma
    s=tf.to_float(s)

    nh=h*s
    nw=w*s
    nh=tf.to_int32(nh)
    nw=tf.to_int32(nw)
    im=tf.image.resize_images(im,[nh,nw])
    im=tf.expand_dims(im,axis=0)
    im=tf.pad(im,[[0,0],[0,size-nh],[0,size-nw],[0,0]],constant_values=127)
    im = im / 255.0
    im.set_shape(tf.TensorShape([None, None,None,3]))

    return im, 1/s,h,w

    pass
def handle_im(im_input, size):
    print(size)
    im = tf.image.decode_jpeg(im_input, channels=3)
    h = tf.shape(im)[0]
    w = tf.shape(im)[1]
    im = tf.image.resize_images(im, [size, size])
    im = tf.to_float(im)
    sh = (h - 1) / (size - 1)
    sw = (w - 1) / (size - 1)
    box_scale = tf.concat([[sw], [sh], [sw], [sh]], axis=0)
    im = im / 255.0
    im = tf.expand_dims(im, axis=0)
    return im, box_scale,h,w
    pass

def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """

    file = open(cfgfile, 'r')
    lines = file.read().split('\n')  # store the lines in a list
    lines = [x for x in lines if len(x) > 0]  # get read of the empty lines
    lines = [x for x in lines if x[0] != '#']  # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":  # This marks the start of a new block
            if len(block) != 0:  # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)  # add it the blocks list
                block = {}  # re-init the block
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


def leaky_relu(net):
    return slim.nn.leaky_relu(net, 0.1)


def conv2d(net, num_outputs, kernel_size, batch_norm, padding, stride, activation, name):
    if padding == 1 and stride==1:
        pad = 'SAME'
    else:
        pad = 'VALID'
        net=tf.pad(net,[[0,0],[1,1],[1,1],[0,0]])
    if activation == 'leaky':
        act = leaky_relu
    else:
        act = None

    if batch_norm == 1:

        net = slim.conv2d(net, num_outputs, [kernel_size, kernel_size], activation_fn=None, stride=stride,
                          biases_initializer=None,
                          padding=pad,
                          scope='conv' + name)
        net = slim.batch_norm(net, activation_fn=act, scope='norm' + name)
    else:
        net = slim.conv2d(net, num_outputs, [kernel_size, kernel_size], padding=pad, stride=stride, activation_fn=act,
                          scope='conv' + name)
    return net


def create_net(net, tblocks):
    blocks=copy.deepcopy(tblocks)
    end_points = []
    Name = []
    for index, x in enumerate(blocks[1:]):
        if (x["type"] == "convolutional"):
            # Get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
            except:
                batch_normalize = 0

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            net = conv2d(net, filters, kernel_size, batch_normalize, padding, stride, activation,
                         str(index))
            end_points.append(net)
            Name.append('conv')
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            H = tf.shape(net)[1]
            W = tf.shape(net)[2]
            net=tf.image.resize_nearest_neighbor(net,[H*2,W*2])


            end_points.append(net)
            Name.append('upsample')
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            # Start  of a route
            start = int(x["layers"][0])
            # end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            if end == 0:
                net = end_points[start]
            else:
                s = end_points[start]
                e = end_points[end]
                net = tf.concat((s,e), axis=-1,name='concat'+str(index))

            end_points.append(net)
            Name.append('route')
        elif x["type"] == "shortcut":
            s = int(x['from'])
            # net = net + end_points[s]
            net=tf.add(net,end_points[s],name='add'+str(index))

            end_points.append(net)
            Name.append('res')
        elif x["type"] == 'reorg':
            net = tf.space_to_depth(net, 2)
            end_points.append(net)
            Name.append('reorg')

        elif x["type"] == 'maxpool':
            size = int(x['size'])
            stride = int(x['stride'])
            net = slim.max_pool2d(net, [size, size], stride=stride)
            end_points.append(net)
            Name.append('max')
        else:
            end_points.append(net)
            Name.append('detection')
    return Name, end_points


def load_weights(file, blocks, Name, vars,out,count=5):
    with codecs.open(file, 'rb') as fp:
        header = np.fromfile(fp, dtype=np.int32, count=count)
        print(header)
        weights = np.fromfile(fp, dtype=np.float32)

    tblocks = blocks[1:]
    j = 0
    ptr = 0
    W = []
    for i in range(len(Name)):
        name = Name[i]
        # end = end_points[i]
        block = tblocks[i]
        if name == 'conv':
            if 'batch_normalize' in block.keys():

                tvars = vars[j:j + 5]

                j += 5
                shapes = [v.shape.as_list() for v in tvars]
                conv_shape = shapes[0]

                a, b, c, d = conv_shape
                num = a * b * c * d

                beta = weights[ptr:ptr + d]
                ptr += d
                gamma = weights[ptr:ptr + d]
                ptr += d
                mean = weights[ptr:ptr + d]
                ptr += d
                vari = weights[ptr:ptr + d]
                ptr += d
                w = weights[ptr:ptr + num]
                ptr += num

                w = w.reshape(d, c, a, b)
                w = np.transpose(w, [2, 3, 1, 0])
                W.append(w)
                W.append(gamma)
                W.append(beta)
                W.append(mean)
                W.append(vari)

                if ptr == weights.shape[0]:
                    break
            else:
                tvars = vars[j:j + 2]

                j += 2
                shapes = [v.shape.as_list() for v in tvars]
                conv_shape = shapes[0]

                a, b, c, d = conv_shape
                d=out
                num = a * b * c * d
                biases = weights[ptr:ptr + d]
                ptr += d
                w = weights[ptr:ptr + num]
                ptr += num

                w = w.reshape(d, c, a, b)
                w = np.transpose(w, [2, 3, 1, 0])
                W.append(w)
                W.append(biases)
                if ptr == weights.shape[0]:
                    break
                pass
    # print('weights:',ptr == weights.shape[0])
    assert ptr == weights.shape[0],'weights uzip error'
    return W


if __name__ == "__main__":
    pass
