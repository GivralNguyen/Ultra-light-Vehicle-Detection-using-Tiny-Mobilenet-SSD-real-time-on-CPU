

# Model
* The person detection model based on SSD architecture
* There are 3 blocks: Backbone, Extralayers, Detection head

## Backbone

* The extractor is used: tiny [Mobilenet-V2]() + [RFB]() component
* The original Mobilenet-V2 can achhive high accuracy (use [pre-trained](https://storage.googleapis.com/models-hao/mb2-ssd-lite-mp-0_686.pth) from VOC dataset). However the running time also high
* The tiny Mobilenet-V2 is customed Mobilenet-V2 (vertical, horizontal).
```
The backbone can be customed at: ./module/rfb_tiny_mobilenet_v2.py
```


## Extra layers
* Multi-scale feature maps for detection
* The SSD architecture uses multiple layers (multi-scale feature maps) to detect objects independently. As CNN reduces the spatial dimension gradually, the resolution of the feature maps also decrease. SSD uses lower resolution layers to detect larger scale objects and vice versa. For example, the 4× 4 feature maps are used for larger scale object.
* Because the model person detection is  used for surveillance camera (detect samll, medium bojects) so that the small feature maps do not contains lots of information. Thus, this model is eliminated 2 last feature maps
* To the person dection use feature maps: 40-30, 20-15, 10-8, 5-4 (width-height). The resolution of feature map depending on the the input of network.

* Note that: The high resolution of input network is not synonymous with the good results
```
The Extra Layers can be customed at: ./model/rfb_tiny_mb_ssd.py and ./module/ssd.py
```

## Detection head

* Regression head (location) and  classification header (classification)
* The most impotant factor in this component is anchor boxes (whith 3 parametert can obtimize scale, ratio, number of anchor per gid cell )
* The anchor boxes are defined in [] based on the COCO, VOC dataset with many objects as well as ratio, size...
* In [], the author proposed formula to generate anchors boxes. However, this formula is designed for many object catagory , size... For instance, the proposed scale factor in [] are [0.2, 0.9] which can be suitable for COCO dataset. However, this factor does not give good results... As mentioned above, the objects in surveillance applivation distribute small, medium and rarely large. Therefor the scale should have distribution at smaller than 0.5
* Ratio: the ration between height/width of objects. statistics to get this factor. Code is available [here]()

```
Anchor boxes: ./model/config/rfb_tiny_mb_ssd_config.py and ./model/rfb_tiny_mb_ssd.py
```

## Loss 
* Localization oss: Use smooth_l1_loss
* Classification loss: Use focal loss instead of CE loss to  to address the issue of the class imbalance problem (person/background)

# Dataset
* There are 2 main dataset: Crowd-Human- 15k and Wider-Person-8k (both full box).
* COCO person (contain noise, fail annotations)
* Cleaned City person, Eure city person: eliminate boxes which has (area box)/(area image) >

# Requirements
* anaconda
* Python-3.6
* Pytorch-1.2
* Torchvision-0.4
* opencv-python-
* pandas

# Training
* Optimizer: SGD, with weight decay: 5e-4, batch size: 32, Number of echop:
* Training with batch size: 32
* Data augmentation:
```
python train.py, type_network rfb_tiny_mb2_ssd, setting base_channel = 64 with ver-1 or base_channel = 32 with ver2
```
# Model parameter and results
* Input: (320, 240)
* Feature maps: 40-30, 20-15, 10-8, 5-4
* Step (shrinkage): 8-8, 16-16, 32-30, 64-60
* Scale (box size): (10, 16, 24), (32, 48), (64, 96), (128, 192, 256)
* Ratios: 2.21, 2.47, 2.73
* base_channel: Ver1(rfb_tiny_mb2_ssd_c64): 32, Ver2(rfb_tiny_mb2_ssd_c32): 64

  
## Pytorch model
  
| input network | parameter | FLOPs | Miss rate |  AP   | Running time |                                                     Model weight                                                      |
| :-----------: | :-------: | :---: | :-------: | :---: | :----------: | :-------------------------------------------------------------------------------------------------------------------: |
| 320x240  (v1) |   3.9 M   | 2.7 G |   7.7%    |  88%  |     48ms     | [Ver1 weight](http://192.168.0.232:8929/tienln4/ai_camera_detector/-/tree/master/app%2Fperson%2Frfb_tiny_mb2_ssd_c64) |
| 320x240  (v2) |   1,2 M   | 0.8G  |  12.5%%   |  84%  |     24ms     | [Ver2 weight](http://192.168.0.232:8929/tienln4/ai_camera_detector/-/tree/master/app%2Fperson%2Frfb_tiny_mb2_ssd_c32) |

## dlc model:
### Ver1 

* dlc and quantized dlc
* Total parameters: 3937476
* Total MACs per inference: 1365M
* Memory Needed to Run: 332.0

### Ver2 
* dlc and quantized dlc
* Total parameters: 1147452
* Total MACs per inference: 369M
* Memory Needed to Run: 171

  