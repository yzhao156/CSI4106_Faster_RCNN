# CSI4106 project (type 3): </br>Object Detection using Faster R-CNN
</br>

## Group number 21
<b>Group members:</b></br>
<pre>
Student Name        Student Number        Email <br>
Wentao Yang         8957505               wyang010@uottawa.ca <br>
Yi Zhao             8650881               yzhao156@uottawa.ca <br>
Junbo Tang          8639271               btang076@uottawa.ca <br>
</pre>


## This is the GitHub link for the Project Final Report, here is an explaination of the arrangement of files in this repository
1. We put all of the code we have commented for understanding the code in "Comments" folder.</br></br>
2. We put all of the files needed to be replaced for the Faster R-CNN implemetation in folder "all files needed to be replaced for the Faster R-CNN implemetation", the directory structure is just the same as the Faster R-CNN implemetation contained in the following reference list, you can assume the folder "all files needed to be replaced for the Faster R-CNN implemetation" is the root path of your implementation.</br></br>
3. We put all of the files needed to be replaced for the Fast R-CNN (Baseline approach) implemetation in folder "all files needed to be replaced for the baseline approach", the directory structure is just the same as the Fast R-CNN implemetation contained in the following reference list, you can assume the folder "all files needed to be replaced for the baseline approach" is the root path of your implementation.</br></br>
4. We also put the files we changed for some of our modified verisons in this repository, including "Wentao's attempt to change the optimizer" and "Yi's attempt to the ResNet and vgg19". Some modifications are so simple to implement (like just changing the learning rate or the max number of epochs in the configuration file) that collecting all of them to here will make this repository a little bit messy.
</br></br>

### Reference
Faster R-CNN:
https://github.com/dBeker/Faster-RCNN-TensorFlow-Python3

Fast R-CNN (Baseline approach):
https://github.com/rbgirshick/fast-rcnn

PASCAL VOC 2007 dataset:
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/

caffe-fast-rcnn:
https://github.com/rbgirshick/py-faster-rcnn

Caffe:
https://github.com/BVLC/caffe.git

A model library:
https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models

The VGG 16 pretrained model we used:
http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz


</br></br>
### File descriptions for the training process
1. CSI4106_Faster_RCNN/train.py Is the main caller
  + `train.train()` </br>
    + get the training set and load it.
  + `train = Train()` </br>
    + create a session
    + Create a network architecture
    + Load weights
    + Loop training step
                    ` rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss = self.net.train_step(sess, blobs, train_op) `
  
------
2. CSI4106_Faster_RCNN/lib/nets/network.py Compute 4 losses
  + loss1: RPN's binary classification (it's an object or background)</br>` rpn_cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))`
  + loss2: RPN's loss of regression (bbox)</br>` rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])`
  + loss3: fully connected network's softmax (20 classification)</br>`cross_entropy = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.reshape(cls_score, [-1, self._num_classes]), labels=label))`
  + loss4: fully connected network's regression (bbox)</br>`  loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)`
   `loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box`

------

3. CSI4106_Faster_RCNN/lib/nets/vgg16.py Build 4 networks
  + __`build_head`: First net__</br>
     + In VGG 16, 3x3 convolution layers would not change the size of feature map</br>`rpn = slim.conv2d(net, 512, [3, 3], trainable=is_training, weights_initializer=initializer, scope="rpn_conv/3x3")`
     + 2x2 pooling changes the size from 2x2 to 1x1, therefore, the number of pooling layers determines the size
     + 5 conv2d layers and 4 pooling layers (0.5*0.5*0.5*0.5 = 1/16)
     + It's not fully-connected, therefore, the input of the model does not have to resize to the same size since CNN can get any size of image

  + __`build_rpn`: Second net__</br>
     + Input is the feature maps from vgg16
     + In general, have 3x3 convolution and get 256 feature maps, at this moment, there isn't any bounding box (bbox).
     + For each point in the feature map, there is an anchor which corresponds to a region of original input(16× larger (because there are 4 pooling layers, thus 1/0.54 = 16))
     + Therefore, each point (anchor) on the feature map is on a square corresponds to a size of 16 times of the original image and find k anchor boxes.
     + For a 3x3 region on conv feature map, we can find size of 3 types (1x1, 2x1 and 1x2) and 3 base sizes of anchor boxes. Therefore, k is 3x3=9 (9 anchor boxes for a point on a conv feature map).
     + Through conv layers and pooling layers, a 600*1000 image can get 256 feature maps with size about 40*60. 
     + In this way, we have around 2000 boxes for each feature map. Then later we can do classification and regression on the boxes.
     + Classification Layer (2*9=18 1x1 conv)
     + For k (= 9) anchor boxes, do binary classification (foreground or background)
     + A 1x1 conv layer gets 2k scores.
     + Regression Layer (4*9=36 1x1 conv)
     + For each box, there are x1, y1, x2, y2. We regression on these points to adjust the size, shape and position of boxes.
     + Since there are k (=9) anchor boxes for each point, we have 4(x1,y1,x2,y2)*k coordinates


  
  + __`build_proposals`: Third net__</br>
     &nbsp; &nbsp; &nbsp; &nbsp; IOU = Intersation of Unit;
     NMS = Non-maximum suppression;
     bbox = boundary box
     + Background:
       + Input is the boxes that generated in RPN and feature map generated in vgg16.
       + At this moment, we map all boxes to original image.
       + Therefore, we can calculate IOU.
       + When IOU > a standard (0.7), it is an object.
       + Each bbox has a probility for an object, for each anchnor, we have 9 bbox with 9 probabilities (of objectiveness) in RPN layer.
     + Reduce bbox:
        + If we have 2000 boxes, it's too many. We need to reduce the number of boxes and only keep boxes that are valuable. There are three methods to reduce the boxes
            1. IOU
                If IOU<0.7, we don't do regression. Stop.
            2. NMS
                If it's an object (>=0.7), IOU is large. There are many boxes around the same object. Use NMS to keep only the boxes with high probabilities that says it is an object.
            3. MAX
                If the boundary box is out of the image, ignore the bbox.
      + As a result, we reduce the number of bboxes from 2000 to 128 (sorted based on the probability of it's an object, and only take the highest 128 bboxes)      
  + __`build_predictions`: Fourth net__</br>
      &nbsp; &nbsp; &nbsp; &nbsp;  Here is a network that's fully connected, and the size of fc6, fc7 are both 4096.</br>&nbsp; &nbsp; &nbsp; &nbsp;`fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')`</br>&nbsp; &nbsp; &nbsp; &nbsp;`fc7 = slim.fully_connected(fc6, 4096, scope='fc7')`</br>
      &nbsp; &nbsp; &nbsp; &nbsp; Using scores and predictions to do classifications and bbox regression.
      + Do Classfication 
        + cls_score is the result classification(21)</br> `cls_score = slim.fully_connected(fc7, self._num_classes, weights_initializer=initializer, trainable=is_training, activation_fn=None, scope='cls_score')`</br>
        + Input the classification score into the fully connected layer</br>`cls_prob = self._softmax_layer(cls_score, "cls_prob")`
      + Do bbox regression(21x4)
        + Already have target, proposal target layer.</br>` bbox_prediction = slim.fully_connected(fc7, self._num_classes * 4, weights_initializer=initializer_bbox, trainable=is_training, activation_fn=None, scope='bbox_pred')`


  
