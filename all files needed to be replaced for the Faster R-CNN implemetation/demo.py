#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
#from nets.resnet_v1 import resnetv1
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer

# csi4106:
import lib.datasets.pascal_voc as pascal_voc_eval
import lib.utils.test as pascal_voc_test
import lib.datasets.factory as pascal_voc_factory

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

# CSI4106:
global imNumber  # used to count the number of output of the same input image

# CSI4106: a function to create the directory to save all of the output images
def mkdir(path):
    # CSI4106: remove spaces at the beginning and the end
    path = path.strip()
    # CSI4106: remove '\' at the end
    path = path.rstrip("\\")
 
    # CSI4106: determine whether the path is exist or not
    isExists = os.path.exists(path)
    if not isExists:
        # CSI4106: if not, create the directory
        os.makedirs(path) 
        print('Directory: ' + path + ' successfully created')
        return True
    else:
        # CSI4106: if the path already exists, do not create the directory again
        print('Directory: ' + path + ' already exists')
        return False
        
        
# CSI4106: modified to save the output image into a folder with its name and bbox number
def vis_detections(im, image_name, directoryName, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    
    # CSI4106:
    # plt.draw()
    global imNumber
    plt.savefig('./' + directoryName + '/' + image_name + '_' + str(imNumber) + '.jpg')
    imNumber += 1


def demo(sess, net, image_name, directoryName):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.FLAGS2["data_dir"], 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.1
    NMS_THRESH = 0.1
    
    
    # CSI4106: modified to save the output image into a folder with its name and bbox number
    global imNumber
    imNumber = 0
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, image_name, directoryName, cls, dets, thresh=CONF_THRESH)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    # model path
    # CSI 4106:
    # demonet = args.demo_net
    demonet = 'vgg16'   ## the pre-trained model you are using,
                        ## please change before you run demo.py


    dataset = args.dataset
    
    # CSI 4106:
    tfmodel = 'default/voc_2007_trainval/default/vgg16_faster_rcnn_iter_50000.ckpt'   ## the file path of the model you want to use,
                                                            ## please change before you run demo.py
    
    # tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default', NETS[demonet][0])

    # if not os.path.isfile(tfmodel + '.meta'):
    #     print(tfmodel)
    #     raise IOError(('{:s} not found.\nDid you download the proper networks from '
    #                    'our server and place them properly?').format(tfmodel + '.meta'))


    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    # elif demonet == 'res101':
        # net = resnetv1(batch_size=1, num_layers=101)
    else:
        raise NotImplementedError

    n_classes = len(CLASSES)
    # create the structure of the net having a certain shape (which depends on the number of classes) 
    net.create_architecture(sess, "TEST", n_classes,
                            tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    im_names = ['000456.jpg', '000457.jpg', '000542.jpg', '001150.jpg',
                '001763.jpg', '004545.jpg']

    # CSI4106:
    imdb = pascal_voc_factory.get_imdb('voc_2007_test')
    # imdb.competition_mode(args.comp_mode)
    pascal_voc_test.test_net(sess, net, imdb, None)


    # CSI4106: obtain the directory that the user want to output the result images
    print('============================================================================')
    print("\nPlease input the folder name you want to output the result images:\n")
    directoryName = input()
    mkdir(directoryName)   
    
    for im_name in im_names:
        # CSI4106:
        print('----------------------------------------------------------------------------')
        print('Demo for data/demo/{}'.format(im_name))
        demo(sess, net, im_name, directoryName)
    # CSI4106:
    # plt.show()