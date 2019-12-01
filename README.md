# Faster_RCNN With Comment
</br>

_All comments starts with_ __'#4106'__

### Video:

+ 0:32:51 怎么加载数据</br>
+ 1:14:40 网络结构 RPN 9 anchnors | 1x1 cnn 18 classification | 1x1 cnn 36 regression</br>
+ 1:44:54 网络结构 Proposal</br>
+ 2:18:00 END</br>

### Modify(Ordered in comments):

1. CSI4106_Faster_RCNN/lib/nets/vgg16.py</br>
2. CSI4106_Faster_RCNN//train.py</br>
3. CSI4106_Faster_RCNN//lib/nets/network.py </br>
4. CSI4106_Faster_RCNN//lib/layer_utils/proposal_layer.py </br>
5. CSI4106_Faster_RCNN//lib/utils/bbox_transform.py</br>
6. CSI4106_Faster_RCNN//lib/datasets/factory.py </br>

### Config(parameters):

+ CSI4106_Faster_RCNN/lib/config/config.py </br>

### core:

+ CSI4106_Faster_RCNN/lib/nets/network.py has 4 losses
  + loss1: RPN's binary classification (it's an object or background)</br>
  + loss2: RPN's loss of regression (bbox)</br>
  + loss3: fully connected network's softmax (20 classification)</br>
  + loss4: fully connected network's regression (bbox)</br>
+ CSI4106_Faster_RCNN/lib/nets/vgg16.py Build networks
  + build_head: First net</br>
     3x3 convolution layers would not change the size of feature map in vgg16</br>
     2x2 pooling change the size from 2x2 to 1x1, therefore, the number of pooling laber determine the size</br>
     5 conv2d layers and 4 pooling layers (0.5*0.5*0.5*0.5 = 1/16)</br>
     it's not fully connected, therefore, the input of the model does not have  to resize to the same size since CNN can get any size of image 
  + build_rpn: Second net</br>
        Input is the feature maps from vgg16</br>
        In general, habe 3x3 convolution and get 256 feature maps, at this moment, there isn't any bound box.</br>
        For each point in the feature map, there is an anchor which correspond a region of original input(16x larger(4 pooling layers  1/0.5^4 = 16))</br>
        Therefore, for each point(anchor) on the feature map, correspoind 16 times of orginal image and find k anchor boxes.</br>
        For a 3x3 region on conv feature map, we can find size of 1x1,2x1,1x2.(3 types) and 3 base size. Therefore, k is 3x3=9. (9 anchor boxes for a point in a conv feature map)</br>
        In the paper, the writer says an 600*1000 image through conv layers and pooling layers can get 256 feature map with size about 40*60. </br>
        In this way, we get a lot of boxes(about 2000 from the paper) for each feature map. We can use classification and regression to the boxes latter.</br>
        Classification Layer(2*9=18 1x1 conv)</br>
        For k(9) anchor boxes, do binary classfication(frontground or background)</br>
        A conv layer with 1x1 and get 2k scores.</br>
        Regression Layer(4*9=36 1x1 conv)</br>
        For each box, there are x1,y1,x2,y2. We regression on these points to adjust the size,shape,position of boxes.</br>
        Since there are k(9) anchor boxes for each point, we have 4(x1,y1,x2,y2)*k coordinates</br>
  + build_proposals
  + build_predictions


  
