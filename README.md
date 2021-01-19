# Face Mask Detector

### Face mask detection with Tensorflow
My "hello tensorflow" project, 
Inspired by [this blog post](https://data-flair.training/blogs/face-mask-detection-with-python/) and [this repo](https://github.com/chandrikadeb7/Face-Mask-Detection)

#### Static detection example:
![static exmaple](https://github.com/itamarco/face-mask-detector/blob/master/assets/publish/detection_result.png?raw=true)

### Tech stack
* Tensorflow (training and detection) 
* OpenCV (face detection and image processing)
* Colab (Interactive runtime for jupyter notebooks)

### Dataset
#### [link to download](https://drive.google.com/drive/folders/1XDte2DL2Mf_hw4NsmGst7QtYoU7sMBVG?usp=sharing)
Taken from [chandrikadeb7 repo](https://github.com/chandrikadeb7/Face-Mask-Detection)

### Model 
#### [Open in jupyter notebook](https://colab.research.google.com/github/itamarco/face-mask-detector/blob/master/face_masks_model.ipynb)
A convolution network consist of these layers:
* Two pairs of Conv and MaxPool layers to extract features from the dataset.
* Flatten and Dropout layer to convert the data in 1D and ensure overfitting.
* Two Dense layers for classification.
![model summary](https://github.com/itamarco/face-mask-detector/blob/master/assets/publish/model_summary.png?raw=true)

### Live webcam detection
```shell
 $ python live-mask-detector.py
```
Requirements:
* tensorflow (or tensorflow-cpu)
* opencv-python
