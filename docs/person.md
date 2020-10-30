
```
rfb_tiny_mb2_ssd: ver_c32 fast; ver_c64 slow (change at /media/ducanh/DATA/tienln/ai_camera/ai_camera_detector/module/rfb_tiny_mobilenet_v2.py 'self.base_channel')
mb2-ssd-lite_f38_person: for small objects

```
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

## Các vấn đề gặp phải
- Kích thước ảnh lớn không đồng nghĩa với việc model sẽ cho kết quả tốt hơn, việc tăng kích thước đầu vào của model giúp model tạo ra các feature maps lớn--> detect các objects bé hơn
- Giảm kích thước model bằng việc giảm độ sâu model (depth chanel) sẽ bị đánh đổi về độ chính xác rất lớn thay vào đó muốn cân đối kích thước mô hình thì thực hiện thay đổi các feature map trong Extra layers, Số lượng anchors box
- Trong quá trính thiết kế anchor boxes, Nếu áp dụng công thức tính scale (Trong code đặt là box_size) trong anchor boxes trong bài báo gốc (SSD) không đem lại kết quả tốt cho một số trường hợp cụ thể (Ví dụ như bài toán detect đầu hay người với object bé). Cụ thể, trong bài báo tác giả đề xuất scale nằm trong dải 0.2 đến 0.9 với số lượng tùy ý (Muốn đạt được kết quả cao thì thiết kế nhiều đồng thời đánh đổi về tốc độ), tuy nhiên dải đấy mang tính tổng quát, vì trong bộ COCO chứa rất ít các objects bé (15x15 pixel). Do đó, Để thực hiện với các object bé cần chọn giải phân bố của scale khác. Không có một nghiên cứu nào đề xuất ra phương pháp chọn scale phù hợp (Lựa chọn theo kinh nghiệm thực tế). Do đó, để lựa chọn dải của scale cần thực hiện: Lựa chọn dải kích thước mà ứng dụng cần nhắm tới (đầu 15x15 pixel) thì min của scale 15/300 (300 là kích thước ảnh đầu vào). Lưu lý không phân bố đều dải này từ 0-1 mà chủ yếu rtập trung khoảng 0-0.5
- Một tham số quan trong nữa trong anchor boxes là ratio: Đối với đầu thì CAM đặt góc nào thì tỉ lệ chiều cao đầu với chiều rộng không thay đổi nhiều 1.3-1.7. tuy nhiên, Bài toán phát hiện người thì phụ nhiều vào góc lắp Camera, người ngồi, đứng, nằm...thì tỉ lệ này dao động rất lớn (config cho mdel nhiềù ratio thì model hoạt động tốt, tuy nhiên bị đánh dổi về tốc độ- trade off lựa chọn 3 tỉ lệ đối với person và 1 đối với head). Để lựa chọn được tỉ lệ phù hợp cần thực hiện thống kê data bằng việc phân cụm các tỉ lệ (code: )
- Khi thực hiện thay dổi dẫn đến thay dổi số lượng anchor thì cũng cần thay đổi các thông số model để maching. Cụ thể như sau:
### Thay dổi bên Extra layer:
- Lựa chọn các kích thước của feature map từ backbone:
```
source_layer_indexes = [GraphPath(7, 'conv', 3),GraphPath(14, 'conv', 3),19,] line 27 at /media/ducanh/DATA/tienln/ai_camera/ai_camera_detector/model/mb_ssd_lite_f38.py
7 hoặc 14: Thể hiện block thứ 7 hoặc 14 trong backbone
'conv', 3: thể thể hiện lấy từ lớp thứ 3 (conv) trong khối đó
Tóm lại: GraphPath(7, 'conv', 3) có nghĩa là feature map được lấy ra từ khối thứ 7 và lớp thứ 3 trong khối đó,GraphPath(14, 'conv', 3) có nghĩa là feature map được lấy ra từ khối thứ 14 và lớp thứ 3 trong khối đó; 19 có nghĩa là lấy lớp conv cuối cùng của block đó. Để xem các layer trong block chỉ cần print(backbone)
```
- Khi thực hiện thay đổ các feature map đãn đến mất khớp (No mactching) trong detection head (regression_headers, classification_headers): cuj thể là biến 'in_channels' trong hai khối regression_headers, classification_headers. Để thực hiện điền đúng cần in backbone để xem chính các output chanel là bao nhiêu để điền đúng ờ regression_headers, classification_headers

### Thay đổi số lượng anchor
- Việc thay đổi số lượng anchor cho mỗi grid trên feature map  (file config). Cần thay đổi trên model để các thông số khớp với nhau: Cụ thể chỉ cần thay đổi giá trị của list 'anchor' line 23 at /media/ducanh/DATA/tienln/ai_camera/ai_camera_detector/model/mb_ssd_lite_f38.py. Giá trị của list chính là số lượng anchor boxes trên mỗi grid của feature map
- Ví dụ: trong file config số lượng anhor cho mỗi grid là 2 thì giá trị  của list 'anchor' line 23 cũng là 2.
### Data
- Đổi với head detection: Data tập trung vào face nên model hoạt động tốt nhất cho face, kém đổi với đầu phía sau và camera đặt góc âm
- Đối với bài toán: Person detection thì data chủ yếu full, thiếú nửa thân.
- Chú ý: Để mdel học tốt trong quá trình trainiing thì cần lọc các ảnh có kích thước quá bé (10 pixel đối với đầu và 15 đối với người)

  