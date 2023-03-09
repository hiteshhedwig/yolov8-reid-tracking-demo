# Person ReID Yolov8+tracking :

## Problem statement :

Build an AI-based system that can re-identify persons in a video sequence/or live webcam that have been temporarily lost due to occlusion or change in appearance. The system should be able to track the persons even when they are partially or completely occluded, and re-identify them when they reappear. The person should keep the same ID even if they leave the frame and come back again. Count the total number of  persons seen in the image.

----

## Weights :

download weights and store them in `weights/*`

```
https://drive.google.com/drive/folders/1jgq5aXcGjv_4H7UEsICV2RdPwFYNILzB?usp=share_link
```

### Run Your Own :

open the file `config.yml`. change parameters accordingly to your test

```
python yolov8_util.py
```



### References :

https://github.com/mikel-brostrom/yolov8_tracking
