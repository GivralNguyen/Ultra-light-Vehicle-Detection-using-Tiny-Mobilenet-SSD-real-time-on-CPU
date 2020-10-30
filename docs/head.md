
```
mb2-ssd-lite_f38: verson for head
```
# Model
* The person detection model based on SSD architecture
* There are 3 blocks: Backbone, Extralayers, Detection head

## Backbone

* The extractor is used: Mobilenet
* The original Mobilenet-V2 can achhive high accuracy (use [pre-trained](https://storage.googleapis.com/models-hao/mb2-ssd-lite-mp-0_686.pth) from VOC dataset). However the running time also high
* The tiny Mobilenet-V2 is customed Mobilenet-V2 (vertical, horizontal).
```
The backbone can be customed at: ./module/mobilent_v2.py
```


## Extra layers
* Multi-scale feature maps for detection
* The SSD architecture uses multiple layers (multi-scale feature maps) to detect objects independently. As CNN reduces the spatial dimension gradually, the resolution of the feature maps also decrease. SSD uses lower resolution layers to detect larger scale objects and vice versa. For example, the 4× 4 feature maps are used for larger scale object.
* Because the model person detection is  used for surveillance camera (detect samll, medium bojects) so that the small feature maps do not contains lots of information. Thus, this model is eliminated 2 last feature maps
* To the head dection use feature maps: 38, 19, 8, 4 . The resolution of feature map depending on the the input of network.

* Note that: The high resolution of input network is not synonymous with the good results (Experiments)
```
The Extra Layers can be customed at: ./model/mb_ssd_lite_f38.py and ./module/ssd.py
```
**Change feature map**

Step_1: file 'mb_ssd_lite_f38_config.py' at line 27, to get feature map from backbone
Step_2: To maching the depth of Extra layers with detection head. change in chanel at 'mb_ssd_lite_f38_config.py' (regression_headers, classification_headers)
for instance:
- feature map size 38, output depth of 192. so that, at line 33 have to assign  in_channels= 192

**Change number of anchor boxes**
Step_1: file 'mb_ssd_lite_f38_config.py', change number of anchor boxes for each grid on feature map
Step_2: To maching the number of anchor boxes: change 'anchors list' line 23


## Detection head

* Regression head (location) and  classification header (classification)
* The most impotant factor in this component is anchor boxes (whith 3 parametert can obtimize scale, ratio, number of anchor per gid cell )
* The anchor boxes are defined in [] based on the COCO, VOC dataset with many objects as well as ratio, size...
* In [], the author proposed formula to generate anchors boxes. However, this formula is designed for many object catagory , size... For instance, the proposed scale factor in [] are [0.2, 0.9] which can be suitable for COCO dataset. However, this factor does not give good results... As mentioned above, the objects in surveillance applivation distribute small, medium and rarely large. Therefor the scale should have distribution at smaller than 0.5
* Ratio: the ration between height/width of objects. statistics to get this factor. Code is available [here]()

```
Anchor boxes: ./model/config/mb_ssd_lite_f38_config.py and ./module/ssd.py
```


## Loss 
* Localization oss: Use smooth_l1_loss
* Classification loss: Use focal loss instead of CE loss to  to address the issue of the class imbalance problem (person/background)

# Dataset
* There are 3 main dataset: Crowd-Human- 15k and Wider-face- 16k, brainwash-11k
* SCUTB (valid)

# Requirements
* anaconda
* Python-3.6
* Pytorch-1.2
* Torchvision-0.4
* opencv-python-
* pandas

# Training
* Optimizer: SGD, with weight decay: 5e-4, batch size: 32, Number of echop:150
* Data augmentation:
```
python train.py, type_network 'mb2-ssd-lite_f38'
```
# Testing
```
Folder image: python detect_imgs.py --net_type <model_path> --test_path <path_dir_image>
video: python live_demo.py --model_path <path_network>
```
# ModelParameter and results
* Input: (300, 300)
* Feature maps: 38-38, 19-19, 10-10, 5-5
* Step (shrinkage): 8, 16, 32, 64
* Scale (box size): (16-32), (32-64),(64,128),(128,256)
* Ratios: 1.7
  
## Pytorch model (mb_ssd_lite_f38_150_193_14)
  
| input network | parameter | FLOPs | Miss rate |  mAP   | Running time |                                      Model weight                                       |
| :-----------: | :-------: | :---: | :-------: | :----: | :----------: | :-------------------------------------------------------------------------------------: |
|    300x300    |   2.7 M   | 1.2 G |    7%     | 90.02% |     39ms     | [Weight](http://192.168.0.232:8929/tienln4/ai_camera_detector/-/tree/master/app%2Fhead) |

## dlc model:
* dlc and quantized dlc
* Total parameters: 2750864
* Memory Needed to Run: 345.9 MiB
* Total MACs per inference: 618M (100%)

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
