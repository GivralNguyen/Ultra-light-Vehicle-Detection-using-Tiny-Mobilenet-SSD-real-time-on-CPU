# README
* This project  implement the head detection model and human detection. More detail about each model can be read at [head detection](http://192.168.0.232:8929/tienln4/ai_camera_detector/-/blob/master/docs/head.md) and [person detection](http://192.168.0.232:8929/tienln4/ai_camera_detector/-/blob/master/docs/person.md)

# Version
- mb2-ssd-lite_f19: original ssd-lite model
- mb2-ssd-lite_f38: for head detection
- mb2-ssd-lite_f38_person: for person detection (small objects)
- rfb_tiny_mb2_ssd: for person detection (there are two sub version: c32 fast and c64 slow)
```
* config(c32 and c64): at line 76 (/media/ducanh/DATA/tienln/ai_camera/ai_camera_detector/module/rfb_tiny_mobilenet_v2.py)   
```