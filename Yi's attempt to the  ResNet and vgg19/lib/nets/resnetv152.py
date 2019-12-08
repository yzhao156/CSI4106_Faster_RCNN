# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
import numpy as np

from lib.nets.network import Network
from tensorflow.python.framework import ops
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers
from lib.config import config as cfg

def resnet_arg_scope(is_training=True,
                     weight_decay=0.001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
  batch_norm_params = {
    # NOTE 'is_training' here does not work because inside resnet it gets reset:
    # https://github.com/tensorflow/models/blob/master/slim/nets/resnet_v1.py#L187
    'is_training': False,
    'decay': batch_norm_decay,
    'epsilon': batch_norm_epsilon,
    'scale': batch_norm_scale,
    'trainable': True, #4106 change to default
    'updates_collections': ops.GraphKeys.UPDATE_OPS
  }

  with arg_scope(
      [slim.conv2d],
      weights_regularizer=regularizers.l2_regularizer(weight_decay),
      weights_initializer=initializers.variance_scaling_initializer(),
      trainable=is_training,
      activation_fn=nn_ops.relu,
      normalizer_fn=layers.batch_norm,
      normalizer_params=batch_norm_params):
    with arg_scope([layers.batch_norm], **batch_norm_params) as arg_sc:
      return arg_sc

class resnetv1(Network):
  def __init__(self, batch_size=1, num_layers=50):
    Network.__init__(self, batch_size=batch_size)
    self._num_layers = num_layers
    self._resnet_scope = 'resnet_v1_%d' % num_layers

  def _crop_pool_layer(self, bottom, rois, name):
    with tf.variable_scope(name) as scope:
      batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
      # Get the normalized coordinates of bboxes
      bottom_shape = tf.shape(bottom)
      height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
      width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
      x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
      y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
      x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
      y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
      # Won't be backpropagated to rois anyway, but to save time
      bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], 1))
      if cfg.RESNET.MAX_POOL:
        pre_pool_size = cfg.POOLING_SIZE * 2
        crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size],
                                         name="crops")
        crops = slim.max_pool2d(crops, [2, 2], padding='SAME')
      else:
        crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [cfg.POOLING_SIZE, cfg.POOLING_SIZE],
                                         name="crops")
    return crops

  # Do the first few layers manually, because 'SAME' padding can behave inconsistently
  # for images of different sizes: sometimes 0, sometimes 1
  def build_base(self):
    with tf.variable_scope(self._resnet_scope, self._resnet_scope):
      net = resnet_utils.conv2d_same(self._image, 64, 7, stride=2, scope='conv1')
      net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
      net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')

    return net


  def build_network(self, sess, is_training=True):
    # select initializers
# if cfg.TRAIN.TRUNCATED: 4106 comment on these
    #   initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
    #   initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
    # else:
    initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
    initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)
    bottleneck = resnet_v1.bottleneck
    # choose different blocks for different number of layers
    if self._num_layers == 50:
      blocks = [
        resnet_utils.Block('block1', bottleneck,
                           [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        resnet_utils.Block('block2', bottleneck,
                           [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        # Use stride-1 for the last conv4 layer
        resnet_utils.Block('block3', bottleneck,
                           [(1024, 256, 1)] * 5 + [(1024, 256, 1)]),
        resnet_utils.Block('block4', bottleneck, [(2048, 512, 1)] * 3)
      ]
    elif self._num_layers == 101:
      blocks = [
        resnet_utils.Block('block1', bottleneck,
                           [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        resnet_utils.Block('block2', bottleneck,
                           [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        # Use stride-1 for the last conv4 layer
        resnet_utils.Block('block3', bottleneck,
                           [(1024, 256, 1)] * 22 + [(1024, 256, 1)]),
        resnet_utils.Block('block4', bottleneck, [(2048, 512, 1)] * 3)
      ]
    elif self._num_layers == 152:
      '''org:
      blocks = [
        resnet_utils.Block('block1', bottleneck,
                           [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        resnet_utils.Block('block2', bottleneck,
                           [(512, 128, 1)] * 7 + [(512, 128, 2)]),
        # Use stride-1 for the last conv4 layer
        resnet_utils.Block('block3', bottleneck,
                           [(1024, 256, 1)] * 35 + [(1024, 256, 1)]),
        resnet_utils.Block('block4', bottleneck, [(2048, 512, 1)] * 3)
      ]
      '''
        depth_func = lambda d: max(int(d * 1), 8) #4106 define a helper func
        scope = 'resnet_v1_152'. # add a scope
		#4106 try build new blocks fit the model(152)

        blocks = [
            resnet_utils.Block(scope, bottleneck, [{
                'depth': depth_func(64) * 4,
                'depth_bottleneck': depth_func(64),
                'stride': 1
            }] * (3 - 1) + [{
                'depth': depth_func(64) * 4,
                'depth_bottleneck': depth_func(64),
                'stride': 2
            }]),

            resnet_utils.Block(scope, bottleneck, [{
                'depth': depth_func(128) * 4,
                'depth_bottleneck': depth_func(128),
                'stride': 1
            }] * (8 - 1) + [{
                'depth': depth_func(128) * 4,
                'depth_bottleneck': depth_func(128),
                'stride': 2
            }]),

            resnet_utils.Block(scope, bottleneck, [{
                'depth': depth_func(256) * 4,
                'depth_bottleneck': depth_func(256),
                'stride': 1
            }] * (36 - 1) + [{
                'depth': depth_func(256) * 4,
                'depth_bottleneck': depth_func(256),
                'stride': 2
            }]),

            resnet_utils.Block(scope, bottleneck, [{
                'depth': depth_func(64) * 4,
                'depth_bottleneck': depth_func(64),
                'stride': 1
            }] * (3 - 1) + [{
                'depth': depth_func(64) * 4,
                'depth_bottleneck': depth_func(512),
                'stride': 1
            }])

        ]
    else:
      # other numbers are not supported
      raise NotImplementedError

    # assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    # if cfg.RESNET.FIXED_BLOCKS == 3:
    #   with slim.arg_scope(resnet_arg_scope(is_training=False)):
    #     net = self.build_base()
    #     net_conv4, _ = resnet_v1.resnet_v1(net,
    #                                        blocks[0:cfg.RESNET.FIXED_BLOCKS],
    #                                        global_pool=False,
    #                                        include_root_block=False,
    #                                        scope=self._resnet_scope)
    # elif cfg.RESNET.FIXED_BLOCKS > 0:
    if True:#4106 again change to default
      with slim.arg_scope(resnet_arg_scope(is_training=False)):
        net = self.build_base()
        net, _ = resnet_v1.resnet_v1(net,
                                     blocks[0:1],#4106 blocks[0:cfg.RESNET.FIXED_BLOCKS],
                                     global_pool=False,
                                     include_root_block=False,
                                     scope=self._resnet_scope)

      with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        net_conv4, _ = resnet_v1.resnet_v1(net,
                                           blocks[1:-1], #4106 blocks[cfg.RESNET.FIXED_BLOCKS:-1],
                                           global_pool=False,
                                           include_root_block=False,
                                           scope=self._resnet_scope)
    else:  # cfg.RESNET.FIXED_BLOCKS == 0
      with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        net = self.build_base()
        net_conv4, _ = resnet_v1.resnet_v1(net,
                                           blocks[0:-1],
                                           global_pool=False,
                                           include_root_block=False,
                                           scope=self._resnet_scope)

    self._act_summaries.append(net_conv4)
    self._layers['head'] = net_conv4
    with tf.variable_scope(self._resnet_scope, self._resnet_scope):
      # build the anchors for the image
      self._anchor_component()

      # rpn
      rpn = slim.conv2d(net_conv4, 512, [3, 3], trainable=is_training, weights_initializer=initializer,
                        scope="rpn_conv/3x3")
      self._act_summaries.append(rpn)
      rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn_cls_score')
      # change it so that the score has 2 as its channel size
      rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
      rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
      rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
      rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
      if is_training:
        rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
        # Try to have a determinestic order for the computing graph, for reproducibility
        with tf.control_dependencies([rpn_labels]):
          rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
      else:
        if cfg.TEST.MODE == 'nms':
          rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        elif cfg.TEST.MODE == 'top':
          rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        else:
          raise NotImplementedError

      # rcnn
      if cfg.POOLING_MODE == 'crop':
        pool5 = self._crop_pool_layer(net_conv4, rois, "pool5")
      else:
        raise NotImplementedError

    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
      fc7, _ = resnet_v1.resnet_v1(pool5,
                                   blocks[-1:],
                                   global_pool=False,
                                   include_root_block=False,
                                   scope=self._resnet_scope)

    with tf.variable_scope(self._resnet_scope, self._resnet_scope):
      # Average pooling done by reduce_mean
      fc7 = tf.reduce_mean(fc7, axis=[1, 2])
      cls_score = slim.fully_connected(fc7, self._num_classes, weights_initializer=initializer,
                                       trainable=is_training, activation_fn=None, scope='cls_score')
      cls_prob = self._softmax_layer(cls_score, "cls_prob")
      bbox_pred = slim.fully_connected(fc7, self._num_classes * 4, weights_initializer=initializer_bbox,
                                       trainable=is_training,
                                       activation_fn=None, scope='bbox_pred')
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
      # exclude the first conv layer to swap RGB to BGR
      if v.name == (self._resnet_scope + '/conv1/weights:0'):
        self._variables_to_fix[v.name] = v
        continue
      if v.name.split(':')[0] in var_keep_dic:
        print('Varibles restored: %s' % v.name)
        variables_to_restore.append(v)

    return variables_to_restore

  def fix_variables(self, sess, pretrained_model):
    print('Fix Resnet V1 layers..')
    with tf.variable_scope('Fix_Resnet_V1') as scope:
      with tf.device("/cpu:0"):
        # fix RGB to BGR
        conv1_rgb = tf.get_variable("conv1_rgb", [7, 7, 3, 64], trainable=False)
        restorer_fc = tf.train.Saver({self._resnet_scope + "/conv1/weights": conv1_rgb})
        restorer_fc.restore(sess, pretrained_model)

        sess.run(tf.assign(self._variables_to_fix[self._resnet_scope + '/conv1/weights:0'],
                           tf.reverse(conv1_rgb, [2])))
''' 
----------------------------------------------------------------
4106
rest of code are from https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py
----------------------------------------------------------------
'''
  @slim.add_arg_scope
  def bottleneck(inputs,
                 depth,
                 depth_bottleneck,
                 stride,
                 rate=1,
                 outputs_collections=None,
                 scope=None,
                 use_bounded_activations=False):
      """Bottleneck residual unit variant with BN after convolutions.
      This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for
      its definition. Note that we use here the bottleneck variant which has an
      extra bottleneck layer.
      When putting together two consecutive ResNet blocks that use this unit, one
      should use stride = 2 in the last unit of the first block.
      Args:
        inputs: A tensor of size [batch, height, width, channels].
        depth: The depth of the ResNet unit output.
        depth_bottleneck: The depth of the bottleneck layers.
        stride: The ResNet unit's stride. Determines the amount of downsampling of
          the units output compared to its input.
        rate: An integer, rate for atrous convolution.
        outputs_collections: Collection to add the ResNet unit output.
        scope: Optional variable_scope.
        use_bounded_activations: Whether or not to use bounded activations. Bounded
          activations better lend themselves to quantized inference.
      Returns:
        The ResNet unit's output.
      """
      with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
          depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
          if depth == depth_in:
              shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
          else:
              shortcut = slim.conv2d(
                  inputs,
                  depth, [1, 1],
                  stride=stride,
                  activation_fn=tf.nn.relu6 if use_bounded_activations else None,
                  scope='shortcut')

          residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1,
                                 scope='conv1')
          residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
                                              rate=rate, scope='conv2')
          residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                                 activation_fn=None, scope='conv3')

          if use_bounded_activations:
              # Use clip_by_value to simulate bandpass activation.
              residual = tf.clip_by_value(residual, -6.0, 6.0)
              output = tf.nn.relu6(shortcut + residual)
          else:
              output = tf.nn.relu(shortcut + residual)

          return slim.utils.collect_named_outputs(outputs_collections,
                                                  sc.name,
                                                  output)

  def resnet_v1(inputs,
                blocks,
                num_classes=None,
                is_training=True,
                global_pool=True,
                output_stride=None,
                include_root_block=True,
                spatial_squeeze=True,
                store_non_strided_activations=False,
                reuse=None,
                scope=None):
      """Generator for v1 ResNet models.
      This function generates a family of ResNet v1 models. See the resnet_v1_*()
      methods for specific model instantiations, obtained by selecting different
      block instantiations that produce ResNets of various depths.
      Training for image classification on Imagenet is usually done with [224, 224]
      inputs, resulting in [7, 7] feature maps at the output of the last ResNet
      block for the ResNets defined in [1] that have nominal stride equal to 32.
      However, for dense prediction tasks we advise that one uses inputs with
      spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
      this case the feature maps at the ResNet output will have spatial shape
      [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
      and corners exactly aligned with the input image corners, which greatly
      facilitates alignment of the features to the image. Using as input [225, 225]
      images results in [8, 8] feature maps at the output of the last ResNet block.
      For dense prediction tasks, the ResNet needs to run in fully-convolutional
      (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all
      have nominal stride equal to 32 and a good choice in FCN mode is to use
      output_stride=16 in order to increase the density of the computed features at
      small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.
      Args:
        inputs: A tensor of size [batch, height_in, width_in, channels].
        blocks: A list of length equal to the number of ResNet blocks. Each element
          is a resnet_utils.Block object describing the units in the block.
        num_classes: Number of predicted classes for classification tasks.
          If 0 or None, we return the features before the logit layer.
        is_training: whether batch_norm layers are in training mode. If this is set
          to None, the callers can specify slim.batch_norm's is_training parameter
          from an outer slim.arg_scope.
        global_pool: If True, we perform global average pooling before computing the
          logits. Set to True for image classification, False for dense prediction.
        output_stride: If None, then the output will be computed at the nominal
          network stride. If output_stride is not None, it specifies the requested
          ratio of input to output spatial resolution.
        include_root_block: If True, include the initial convolution followed by
          max-pooling, if False excludes it.
        spatial_squeeze: if True, logits is of shape [B, C], if false logits is
            of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
            To use this parameter, the input images must be smaller than 300x300
            pixels, in which case the output logit layer does not contain spatial
            information and can be removed.
        store_non_strided_activations: If True, we compute non-strided (undecimated)
          activations at the last unit of each block and store them in the
          `outputs_collections` before subsampling them. This gives us access to
          higher resolution intermediate activations which are useful in some
          dense prediction problems but increases 4x the computation and memory cost
          at the last unit of each block.
        reuse: whether or not the network and its variables should be reused. To be
          able to reuse 'scope' must be given.
        scope: Optional variable_scope.
      Returns:
        net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
          If global_pool is False, then height_out and width_out are reduced by a
          factor of output_stride compared to the respective height_in and width_in,
          else both height_out and width_out equal one. If num_classes is 0 or None,
          then net is the output of the last ResNet block, potentially after global
          average pooling. If num_classes a non-zero integer, net contains the
          pre-softmax activations.
        end_points: A dictionary from components of the network to the corresponding
          activation.
      Raises:
        ValueError: If the target output_stride is not valid.
      """
      with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
          end_points_collection = sc.original_name_scope + '_end_points'
          with slim.arg_scope([slim.conv2d, bottleneck,
                               resnet_utils.stack_blocks_dense],
                              outputs_collections=end_points_collection):
              with (slim.arg_scope([slim.batch_norm], is_training=is_training)
              if is_training is not None else NoOpScope()):
                  net = inputs
                  if include_root_block:
                      if output_stride is not None:
                          if output_stride % 4 != 0:
                              raise ValueError('The output_stride needs to be a multiple of 4.')
                          output_stride /= 4
                      net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
                      net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
                  net = resnet_utils.stack_blocks_dense(net, blocks, output_stride,
                                                        store_non_strided_activations)
                  # Convert end_points_collection into a dictionary of end_points.
                  end_points = slim.utils.convert_collection_to_dict(
                      end_points_collection)

                  if global_pool:
                      # Global average pooling.
                      net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
                      end_points['global_pool'] = net
                  if num_classes:
                      net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                        normalizer_fn=None, scope='logits')
                      end_points[sc.name + '/logits'] = net
                      if spatial_squeeze:
                          net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
                          end_points[sc.name + '/spatial_squeeze'] = net
                      end_points['predictions'] = slim.softmax(net, scope='predictions')
                  return net, end_points

  resnet_v1.default_image_size = 224

  def resnet_v1_152(inputs,
                    num_classes=None,
                    is_training=True,
                    global_pool=True,
                    output_stride=None,
                    store_non_strided_activations=False,
                    spatial_squeeze=True,
                    min_base_depth=8,
                    depth_multiplier=1,
                    reuse=None,
                    scope='resnet_v1_152'):
      """ResNet-152 model of [1]. See resnet_v1() for arg and return description."""
      depth_func = lambda d: max(int(d * depth_multiplier), min_base_depth)
      blocks = [
          resnet_v1_block('block1', base_depth=depth_func(64), num_units=3,
                          stride=2),
          resnet_v1_block('block2', base_depth=depth_func(128), num_units=8,
                          stride=2),
          resnet_v1_block('block3', base_depth=depth_func(256), num_units=36,
                          stride=2),
          resnet_v1_block('block4', base_depth=depth_func(512), num_units=3,
                          stride=1),
      ]
      return resnet_v1(inputs, blocks, num_classes, is_training,
                       global_pool=global_pool, output_stride=output_stride,
                       include_root_block=True, spatial_squeeze=spatial_squeeze,
                       store_non_strided_activations=store_non_strided_activations,
                       reuse=reuse, scope=scope)

  def resnet_v1_block(scope, base_depth, num_units, stride):
      """Helper function for creating a resnet_v1 bottleneck block.
      Args:
        scope: The scope of the block.
        base_depth: The depth of the bottleneck layer for each unit.
        num_units: The number of units in the block.
        stride: The stride of the block, implemented as a stride in the last unit.
          All other units have stride=1.
      Returns:
        A resnet_v1 bottleneck block.
      """
      return resnet_utils.Block(scope, bottleneck, [{
          'depth': base_depth * 4,
          'depth_bottleneck': base_depth,
          'stride': 1
      }] * (num_units - 1) + [{
          'depth': base_depth * 4,
          'depth_bottleneck': base_depth,
          'stride': stride
      }])


  def resnet_v1_50(inputs,
                   num_classes=None,
                   is_training=True,
                   global_pool=True,
                   output_stride=None,
                   spatial_squeeze=True,
                   store_non_strided_activations=False,
                   min_base_depth=8,
                   depth_multiplier=1,
                   reuse=None,
                   scope='resnet_v1_50'):
      """ResNet-50 model of [1]. See resnet_v1() for arg and return description."""
      depth_func = lambda d: max(int(d * depth_multiplier), min_base_depth)
      blocks = [
          resnet_v1_block('block1', base_depth=depth_func(64), num_units=3,
                          stride=2),
          resnet_v1_block('block2', base_depth=depth_func(128), num_units=4,
                          stride=2),
          resnet_v1_block('block3', base_depth=depth_func(256), num_units=6,
                          stride=2),
          resnet_v1_block('block4', base_depth=depth_func(512), num_units=3,
                          stride=1),
      ]
      return resnet_v1(inputs, blocks, num_classes, is_training,
                       global_pool=global_pool, output_stride=output_stride,
                       include_root_block=True, spatial_squeeze=spatial_squeeze,
                       store_non_strided_activations=store_non_strided_activations,
                       reuse=reuse, scope=scope)

  # resnet_v1_50.default_image_size = resnet_v1.default_image_size

  def resnet_v1_101(inputs,
                    num_classes=None,
                    is_training=True,
                    global_pool=True,
                    output_stride=None,
                    spatial_squeeze=True,
                    store_non_strided_activations=False,
                    min_base_depth=8,
                    depth_multiplier=1,
                    reuse=None,
                    scope='resnet_v1_101'):
      """ResNet-101 model of [1]. See resnet_v1() for arg and return description."""
      depth_func = lambda d: max(int(d * depth_multiplier), min_base_depth)
      blocks = [
          resnet_v1_block('block1', base_depth=depth_func(64), num_units=3,
                          stride=2),
          resnet_v1_block('block2', base_depth=depth_func(128), num_units=4,
                          stride=2),
          resnet_v1_block('block3', base_depth=depth_func(256), num_units=23,
                          stride=2),
          resnet_v1_block('block4', base_depth=depth_func(512), num_units=3,
                          stride=1),
      ]
      return resnet_v1(inputs, blocks, num_classes, is_training,
                       global_pool=global_pool, output_stride=output_stride,
                       include_root_block=True, spatial_squeeze=spatial_squeeze,
                       store_non_strided_activations=store_non_strided_activations,
                       reuse=reuse, scope=scope)