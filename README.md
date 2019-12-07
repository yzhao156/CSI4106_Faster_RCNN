# Faster_RCNN With Comment
</br>

_All comments in the code starts with_ __'#4106'__</br>
Packages are used: numpy, PIL, scipy, xml.etree.ElementTree , uuid, subprocess, pickle, and tensorflow.</br>
Mathine to train: 3 x NVIDIA Tesla V100 from Google Cloud Platform(GCP)

### Comments(Ordered in the lines of comments):

1. CSI4106_Faster_RCNN/lib/nets/vgg16.py</br>
2. CSI4106_Faster_RCNN//train.py</br>
3. CSI4106_Faster_RCNN//lib/nets/network.py </br>
4. CSI4106_Faster_RCNN/lib/datasets/pascal_voc.py </br>
5. CSI4106_Faster_RCNN//lib/layer_utils/proposal_layer.py </br>
6. CSI4106_Faster_RCNN//lib/utils/bbox_transform.py</br>
7. CSI4106_Faster_RCNN//lib/datasets/factory.py </br>

### Config(parameters):

+ CSI4106_Faster_RCNN/lib/config/config.py </br>
### Core Concept:
1. Pre train a CNN network(vgg16,vgg19,and tried resnet 152)
2. Train the RPN's region proposal first time, and initialized by the pre-train image classifier. Tag samples have intersection over union > 0.7 as positive samples.
  + Use sliding window of small nxn on the conv feature map(not the pixels) of the image.
  + Then predict multiple regions of various scales and ratios simultaneously. An anchor is a conbination of sliding window center, scale, ratio).</br>
  In this implemenation, 3 scales(1x1,2x1,1x2)+ 3 base size(ratios) which is 9 anchors for each sliding position.
3. Train Faster RCNN using the proposals generated by the current RPN
4. Use Faster RCNN network to initialize RPN training. Keeping the shared conv layers. At this stage, RPN and the dection network have shared convolutional layers.
5. Finally fine-tune the unique layers of Fast R-CNN<br>(Step 4-5 can be repeated to train RPN and Fast R-CNN alternatively if needed.)

### Core Files:
1. CSI4106_Faster_RCNN/train.py Is the main caller
  + `train.train()` </br>
    + get the training set and load it.
  + `train = Train()` </br>
    + create a session
    + Create a network architecture
    + Load weights
    + Loop train step
                    ` rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss = self.net.train_step(sess, blobs, train_op) `
  
------
2. CSI4106_Faster_RCNN/lib/nets/network.py Compute 4 losses
  + loss1: RPN's binary classification (it's an object or background)</br>` rpn_cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))`
  + loss2: RPN's loss of regression (bbox)</br>` rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])`
  + loss3: fully connected network's softmax (20 classification)</br>`cross_entropy = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.reshape(cls_score, [-1, self._num_classes]), labels=label))`
  + loss4: fully connected network's regression (bbox)</br>`  loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)`
  + `loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box`

------

3. CSI4106_Faster_RCNN/lib/nets/vgg16.py Build 4 networks
  + __`build_head`: First net__</br>
     + 3x3 convolution layers would not change the size of feature map in vgg16</br>`rpn = slim.conv2d(net, 512, [3, 3], trainable=is_training, weights_initializer=initializer, scope="rpn_conv/3x3")`
     + 2x2 pooling change the size from 2x2 to 1x1, therefore, the number of pooling laber determine the size</br>
     + 5 conv2d layers and 4 pooling layers (0.5*0.5*0.5*0.5 = 1/16)</br>
     + It's not fully connected, therefore, the input of the model does not have  to resize to the same size since CNN can get any size of image 
  + __`build_rpn`: Second net__</br>
     + Input is the feature maps from vgg16</br>
     + In general, habe 3x3 convolution and get 256 feature maps, at this moment, there isn't any bound box.</br>
     + For each point in the feature map, there is an anchor which correspond a region of original input(16x larger(4 pooling layers  1/0.5^4 = 16))</br>
     + Therefore, for each point(anchor) on the feature map, correspoind 16 times of orginal image and find k anchor boxes.</br>
     + For a 3x3 region on conv feature map, we can find size of 1x1,2x1,1x2.(3 types) and 3 base size. Therefore, k is 3x3=9. (9 anchor boxes for a point in a conv feature map)</br>
     + In the paper, the writer says an 600*1000 image through conv layers and pooling layers can get 256 feature map with size about 40*60. </br>
     + In this way, we get a lot of boxes(about 2000 from the paper) for each feature map. We can use classification and regression to the boxes latter.</br>
     + Classification Layer(2*9=18 1x1 conv)</br>
     + For k(9) anchor boxes, do binary classfication(frontground or background)</br>
     + A conv layer with 1x1 and get 2k scores.</br>
     + Regression Layer(4*9=36 1x1 conv)</br>
     + For each box, there are x1,y1,x2,y2. We regression on these points to adjust the size,shape,position of boxes.</br>
     + Since there are k(9) anchor boxes for each point, we have 4(x1,y1,x2,y2)*k coordinates</br>
  
  + __`build_proposals`: Third net__</br>
     &nbsp; &nbsp; &nbsp; &nbsp; IOU = Intersation of Unit;
     NMS = Non-maximum suppression;
     bbox = boundary box
     + Background:
       + Input is the boxes that generated in RPN and feature map generated in vgg16.
       + At this moment, we map all boxes to original image.
       + Therefore, we can calculate IOU.
       + When IOU > a standard(0.7), it is an object.
       + Each bbox has a probility for an object, for each anchnor, we have 9 bbox with 9 probility(of objectiveness) in RPN layer.
     + Reduce bbox:
        + If we have 2000 boxes, it's a lot. We reduce the number of boxes and only keep boxes that are valuable. There are three methods to reduce the boxes
            1. IOU
                If IOU<0.7, we don't do regression. Stop.
            2. NMS
                If it's an object(>=0.7), IOU is large. There are many boxes around the same object. 
                Use NMS to keep only the boxes with high probility of it's an object.
            3. MAX
                If the boundary box is out of the image, ignore the bbox.
      + As a result, we reduce bbox from 2000 to 128(sort based on the probility of it's an object, and only take the highest 128 bbox)        
  + __`build_predictions`: Fourth net__</br>
      &nbsp; &nbsp; &nbsp; &nbsp; Here is a network that's fully connected net and fc6, fc7 with size of 4096.</br>&nbsp; &nbsp; &nbsp; &nbsp;`fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')`</br>&nbsp; &nbsp; &nbsp; &nbsp;`fc7 = slim.fully_connected(fc6, 4096, scope='fc7')`</br>
      &nbsp; &nbsp; &nbsp; &nbsp; Use scores and predictions to do classfications and bbox regression.
      + Do Classfication 
        + cls_score is the result classification(21)</br> `cls_score = slim.fully_connected(fc7, self._num_classes, weights_initializer=initializer, trainable=is_training, activation_fn=None, scope='cls_score')`</br>
        + nput the classification score into the fully connected layer</br>`cls_prob = self._softmax_layer(cls_score, "cls_prob")`
      + Do bbox regression(21x4)
        + Already have target, proposal target layer.</br>` bbox_prediction = slim.fully_connected(fc7, self._num_classes * 4, weights_initializer=initializer_bbox, trainable=is_training, activation_fn=None, scope='bbox_pred')`


  
