# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------


import tensorflow as tf
import tensorflow.contrib.slim as slim

import lib.config.config as cfg
from lib.nets.network import Network

class vgg16(Network): # extends Network
    def __init__(self, batch_size=1):
        Network.__init__(self, batch_size=batch_size)

    def build_network(self, sess, is_training=True):
        with tf.variable_scope('vgg_16', 'vgg_16'): 
            #4106 Since vgg16 is a network based on image regonization based on a large dataset.
            #4106 We are doing classfication(object dection), so just use the checkpoint in /imagenet_weights/vgg16.ckpt.
            #4106 In this way, we don't have to train vgg16.


            # select initializer
            if cfg.FLAGS.initializer == "truncated":
                initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
                initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
            else:
                initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
                initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

            # Build head
            #4106 5 CNN layers and 4 Pooling layers
            net = self.build_head(is_training)

            #4106 Build RPN
            #buld rpn(Core of Faster-RCNN)
            #generate 9 boxes for each point in the conv feature map
            #for each point do 36 regression(adjust the position,shape,size) of boxes and 18 classification(frontground or background)
            rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, rpn_cls_score_reshape = self.build_rpn(net, is_training, initializer)

            #4106 Build proposals
            #find 128 most valuable boxes 
            #use IOU,NMS,MAX
            #Build predictions
            rois = self.build_proposals(is_training, rpn_cls_prob, rpn_bbox_pred, rpn_cls_score)

            #4106 Build predictions
            #Having a network that's fully connected,
            #1) Classification(21)
            #       return the result of classification in object among 20 objects and background
            #2) Regression(21x4)
            #       do regression to find the boundary box and return the result
            cls_score, cls_prob, bbox_pred = self.build_predictions(net, rois, is_training, initializer, initializer_bbox)

            #4106 save the results
            self._predictions["rpn_cls_score"] = rpn_cls_score
            self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
            self._predictions["rpn_cls_prob"] = rpn_cls_prob
            self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
            self._predictions["cls_score"] = cls_score
            self._predictions["cls_prob"] = cls_prob
            self._predictions["bbox_pred"] = bbox_pred
            self._predictions["rois"] = rois

            self._score_summaries.update(self._predictions)

            return rois, cls_prob, bbox_pred

    def get_variables_to_restore(self, variables, var_keep_dic):
        variables_to_restore = []

        for v in variables:
            # exclude the conv weights that are fc weights in vgg16
            if v.name == 'vgg_16/fc6/weights:0' or v.name == 'vgg_16/fc7/weights:0':
                self._variables_to_fix[v.name] = v
                continue
            # exclude the first conv layer to swap RGB to BGR
            if v.name == 'vgg_16/conv1/conv1_1/weights:0':
                self._variables_to_fix[v.name] = v
                continue
            if v.name.split(':')[0] in var_keep_dic:
                print('Variables restored: %s' % v.name)
                variables_to_restore.append(v)

        return variables_to_restore

    def fix_variables(self, sess, pretrained_model):
        print('Fix VGG16 layers..')
        with tf.variable_scope('Fix_VGG16'):
            with tf.device("/cpu:0"):
                # fix the vgg16 issue from conv weights to fc weights
                # fix RGB to BGR
                fc6_conv = tf.get_variable("fc6_conv", [7, 7, 512, 4096], trainable=False)
                fc7_conv = tf.get_variable("fc7_conv", [1, 1, 4096, 4096], trainable=False)
                conv1_rgb = tf.get_variable("conv1_rgb", [3, 3, 3, 64], trainable=False)
                restorer_fc = tf.train.Saver({"vgg_16/fc6/weights": fc6_conv,
                                              "vgg_16/fc7/weights": fc7_conv,
                                              "vgg_16/conv1/conv1_1/weights": conv1_rgb})
                restorer_fc.restore(sess, pretrained_model)

                sess.run(tf.assign(self._variables_to_fix['vgg_16/fc6/weights:0'], tf.reshape(fc6_conv,
                                                                                              self._variables_to_fix['vgg_16/fc6/weights:0'].get_shape())))
                sess.run(tf.assign(self._variables_to_fix['vgg_16/fc7/weights:0'], tf.reshape(fc7_conv,
                                                                                              self._variables_to_fix['vgg_16/fc7/weights:0'].get_shape())))
                sess.run(tf.assign(self._variables_to_fix['vgg_16/conv1/conv1_1/weights:0'],
                                   tf.reverse(conv1_rgb, [2])))

    def build_head(self, is_training):
        #4106 3x3 convolution layers would not change the size of feature map in vgg16
        #4106 2x2 pooling change the size from 2x2 to 1x1, therefore, the number of pooling laber determine the size
        #4106 5 conv2d layers and 4 pooling layers (0.5*0.5*0.5*0.5 = 1/16)
        #4106 it's not fully connected, therefore, the input of the model does not have  to resize to the same size since CNN can get any size of image 
        # Main network
        # Layer  1
        net = slim.repeat(self._image, 2, slim.conv2d, 64, [3, 3], trainable=False, scope='conv1')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')

        # Layer 2
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], trainable=False, scope='conv2')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')

        # Layer 3
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], trainable=is_training, scope='conv3')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')

        # Layer 4
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=is_training, scope='conv4')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')

        # Layer 5
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=is_training, scope='conv5')

        # Append network to summaries
        self._act_summaries.append(net)

        # Append network as head layer
        #4106 The first net of Faster-RCNN
        self._layers['head'] = net

        return net

    def build_rpn(self, net, is_training, initializer):
        '''
        4106
        Input is the feature maps from vgg16
        In general, habe 3x3 convolution and get 256 feature maps, at this moment, there isn't any bound box
        for each point in the feature map, there is an anchor which correspond a region of original input(16x larger(4 pooling layers  1/0.5^4 = 16))
        Therefore, for each point(anchor) on the feature map, correspoind 16 times of orginal image and find k anchor boxes.

        For a 3x3 region on conv feature map, we can find size of 1x1,2x1,1x2.(3 types) and 3 base size. Therefore, k is 3x3=9. (9 anchor boxes for a point in a conv feature map)
        In the paper, the writer says an 600*1000 image through conv layers and pooling layers can get 256 feature map with size about 40*60. 
        In this way, we get a lot of boxes(about 2000 from the paper) for each feature map. We can use classification and regression to the boxes latter.

        Classification Layer(2*9=18 1x1 conv)
        For k(9) anchor boxes, do binary classfication(frontground or background)
        A conv layer with 1x1 and get 2k scores.

        Regression Layer(4*9=36 1x1 conv)
        For each box, there are x1,y1,x2,y2. We regression on these points to adjust the size,shape,position of boxes.
        Since there are k(9) anchor boxes for each point, we have 4(x1,y1,x2,y2)*k coordinates

        '''

        
        # Build anchor component
        #4106 generate 9 boxes for each point on the feature map
        #4106 _num_anchors = 9
        self._anchor_component()
        # Create RPN Layer
        rpn = slim.conv2d(net, 512, [3, 3], trainable=is_training, weights_initializer=initializer, scope="rpn_conv/3x3")
        self._act_summaries.append(rpn)

        #4106 Classification
        #4106 2(front or back) * 9(boxes) =18 classifications
        rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training, weights_initializer=initializer, padding='VALID', activation_fn=None, scope='rpn_cls_score')
        # Change it so that the score has 2 as its channel size
        rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
        #4106 softmax on the 18 results
        rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
        #4106 convert to probility
        rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")

        #4106 Regression
        #4106 4(coordinates) * 9(boxes) = 36 regressions
        rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training, weights_initializer=initializer, padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
        return rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, rpn_cls_score_reshape

    def build_proposals(self, is_training, rpn_cls_prob, rpn_bbox_pred, rpn_cls_score):
        '''
        4106
        IOU = Intersation of Unit
        NMS = Non-maximum suppression
        bbox = boundary box

        Background:
        Input is the boxes that generated in RPN and feature map generated in vgg16.
        At this moment, we map all boxes to original image.
        Therefore, we can calculate IOU.
        When IOU > a standard(0.7), it is an object.
        Each bbox has a probility for an object, for each anchnor, we have 9 bbox with 9 probility(of objectiveness) in RPN layer.

        Reduce bbox:
        If we have 2000 boxes, it's a lot. We reduce the number of boxes and only keep boxes that are valuable.
        1. IOU
            If IOU<0.7, we don't do regression. Stop.
        2. NMS
            If it's an object(>=0.7), IOU is large. There are many boxes around the same object. 
            Use NMS to keep only the boxes with high probility of it's an object.
        3. MAX
            If the boundary box is out of the image, ignore the bbox.
        As a result, we reduce bbox from 2000 to 128(sort based on the probility of it's an object, and only take the highest 128 bbox)        
        '''
        if is_training:
            #4106 rpn_cls_prob is the probility of it's an object(From RPN Layer)
            #4106 rpn_bbox_pred is the position of boxes(From RPN Layer)
            #4106 here the _proposal_layer do transforms and find dx dy dw dh for loss function
            rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")

            # Try to have a deterministic order for the computing graph, for reproducibility
            with tf.control_dependencies([rpn_labels]):
                rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
        else:
            if cfg.FLAGS.test_mode == 'nms':
                rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            elif cfg.FLAGS.test_mode == 'top':
                rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            else:
                raise NotImplementedError
        return rois

    def build_predictions(self, net, rois, is_training, initializer, initializer_bbox):
        #4106 we want to build a network that's fully connected
        #fc6, fc7 with size of 4096
        # Crop image ROIs
        pool5 = self._crop_pool_layer(net, rois, "pool5")
        pool5_flat = slim.flatten(pool5, scope='flatten')

        # Fully connected layers
        fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
        if is_training:
            fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True, scope='dropout6')

        fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
        if is_training:
            fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True, scope='dropout7')

        # Scores and predictions

        #4106 Do Classfication
        #4106 cls_score is the result classification(21)
        cls_score = slim.fully_connected(fc7, self._num_classes, weights_initializer=initializer, trainable=is_training, activation_fn=None, scope='cls_score')
        #4106 input the classification score into the fully connected layer
        cls_prob = self._softmax_layer(cls_score, "cls_prob")

        #4106 Do bbox regression(21x4)
        #4106 Already have target, proposal target layer.
        bbox_prediction = slim.fully_connected(fc7, self._num_classes * 4, weights_initializer=initializer_bbox, trainable=is_training, activation_fn=None, scope='bbox_pred')

        return cls_score, cls_prob, bbox_prediction
