# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from lib.utils.bbox_transform import bbox_transform_inv, clip_boxes
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms


def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
    """A simplified version compared to fast/er RCNN
       For details please see the technical report
    """
    if type(cfg_key) == bytes:
        cfg_key = cfg_key.decode('utf-8')

    if cfg_key == "TRAIN":
        #4106 standard of NMS
        pre_nms_topN = cfg.FLAGS.rpn_train_pre_nms_top_n
        post_nms_topN = cfg.FLAGS.rpn_train_post_nms_top_n
        nms_thresh = cfg.FLAGS.rpn_train_nms_thresh
    else:
        pre_nms_topN = cfg.FLAGS.rpn_test_pre_nms_top_n
        post_nms_topN = cfg.FLAGS.rpn_test_post_nms_top_n
        nms_thresh = cfg.FLAGS.rpn_test_nms_thresh

    im_info = im_info[0]
    # Get the scores and bounding boxes
    #4106 scores is the probility of it's an object(From RPN Layer)
    #4106 rpn_bbox_pred is the position of boxes(From RPN Layer)
    scores = rpn_cls_prob[:, :, :, num_anchors:]
    rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))

    scores = scores.reshape((-1, 1))
    proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
    #4106 clip_boxes to check if the box is out of the image.
    proposals = clip_boxes(proposals, im_info[:2])

    # Pick the top region proposals
    #4106 reduce bbox from 2000 to 128(sort based on the probility of it's an object, and only take the highest 128 bbox)
    order = scores.ravel().argsort()[::-1]
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
    proposals = proposals[order, :]
    scores = scores[order]

    # Non-maximal suppression
    keep = nms(np.hstack((proposals, scores)), nms_thresh)

    #4106 Pick th top region proposals after NMS
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    scores = scores[keep]

    # Only support single image as input
    batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

    return blob, scores
