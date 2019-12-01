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

  
